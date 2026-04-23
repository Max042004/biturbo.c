/*
 * biturbo chat mode — interactive REPL or piped single-shot.
 *
 * Flow per turn:
 *   1. Read user line (tty REPL) or all of stdin (pipe).
 *   2. Apply prompt template ("%s" spliced into user text).
 *   3. Encode to token IDs. BOS only on the very first turn; subsequent
 *      turns continue the conversation at the current `pos`.
 *   4. Prefill: feed every prompt token through bt_forward(model, tok, pos)
 *      advancing pos each step. (No batched prefill in biturbo — transformer
 *      decode is one-token-at-a-time.)
 *   5. Sample next token from logits, decode, stream to stdout. Stop on
 *      eos_id / eot_id, on a user-supplied stop token, on the configured
 *      max_tokens, or on degenerate repetition.
 *   6. Loop back to step 1 until /quit, EOF, or Ctrl-C.
 */

#define _POSIX_C_SOURCE 200809L

#include "chat.h"

#include <ctype.h>
#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <unistd.h>

#define MAX_PROMPT_TOKENS 2048
#define LINE_BUF_SIZE     4096

/* ---- Signal handling ------------------------------------------------- */

static volatile sig_atomic_t g_interrupted = 0;

static void sigint_handler(int sig) {
    (void)sig;
    if (g_interrupted) {
        const char msg[] = "\n[interrupted]\n";
        (void)!write(STDERR_FILENO, msg, sizeof(msg) - 1);
        _exit(1);
    }
    g_interrupted = 1;
}

/* ---- Prompt template ------------------------------------------------- */

/* Strip leading whitespace and the common "Q:" / "Question:" / "User:"
 * prefixes so templates don't double-wrap user input. */
static const char *strip_prefix(const char *s) {
    while (*s == ' ' || *s == '\t') s++;
    if (!strncasecmp(s, "question:", 9)) { s += 9; while (*s == ' ') s++; }
    else if (!strncasecmp(s, "user:", 5)) { s += 5; while (*s == ' ') s++; }
    else if ((s[0] == 'Q' || s[0] == 'q') && s[1] == ':') {
        s += 2; while (*s == ' ') s++;
    }
    return s;
}

/* Splice user input into `tpl` at the single "%s" marker. Manual splice
 * rather than snprintf, so a user-supplied template can't trigger format
 * string UB. Returns malloc'd string. */
static char *apply_template(const char *input, const char *tpl) {
    if (!tpl) return strdup(input);

    const char *marker = strstr(tpl, "%s");
    if (!marker) {
        fprintf(stderr, "[chat] warning: template has no %%s placeholder\n");
        return strdup(input);
    }
    if (strstr(marker + 2, "%s")) {
        fprintf(stderr, "[chat] warning: template has multiple %%s\n");
        return strdup(input);
    }

    input = strip_prefix(input);

    size_t pre = (size_t)(marker - tpl);
    size_t mid = strlen(input);
    size_t suf = strlen(marker + 2);
    char *out = (char *)malloc(pre + mid + suf + 1);
    if (!out) return strdup(input);
    memcpy(out, tpl, pre);
    memcpy(out + pre, input, mid);
    memcpy(out + pre + mid, marker + 2, suf);
    out[pre + mid + suf] = '\0';
    return out;
}

/* ---- Stdin helpers --------------------------------------------------- */

/* Read all of stdin into a malloc'd NUL-terminated buffer, stripping
 * trailing whitespace. Returns NULL when the input is empty or on error. */
static char *read_stdin_all(void) {
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) return NULL;

    for (;;) {
        if (cap - len - 1 == 0) {
            size_t new_cap = cap * 2;
            char *nb = (char *)realloc(buf, new_cap);
            if (!nb) { free(buf); return NULL; }
            buf = nb; cap = new_cap;
        }
        size_t n = fread(buf + len, 1, cap - len - 1, stdin);
        len += n;
        if (n == 0) {
            if (ferror(stdin)) { free(buf); return NULL; }
            break;
        }
    }
    buf[len] = '\0';
    while (len > 0) {
        char c = buf[len - 1];
        if (c != '\n' && c != '\r' && c != ' ' && c != '\t') break;
        buf[--len] = '\0';
    }
    if (len == 0) { free(buf); return NULL; }
    return buf;
}

/* Read one line from stdin with a prompt. Returns malloc'd line (trimmed
 * of trailing newline) or NULL on EOF. */
static char *read_line_tty(const char *prompt) {
    fputs(prompt, stderr);
    fflush(stderr);

    char buf[LINE_BUF_SIZE];
    if (!fgets(buf, sizeof(buf), stdin)) return NULL;

    size_t len = strlen(buf);
    while (len > 0 && (buf[len - 1] == '\n' || buf[len - 1] == '\r'))
        buf[--len] = '\0';
    return strdup(buf);
}

/* ---- KV cache reset -------------------------------------------------- */

void chat_reset(bt_model_t *model) {
    (void)model; /* nothing to free — we just rewind `pos` in the caller */
}

/* ---- Generation ------------------------------------------------------ */

typedef struct {
    int prefill_tokens;
    int generated_tokens;
    double prefill_sec;
    double decode_sec;
} turn_stats_t;

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Feed `prompt` into the model, then sample up to max_tokens. Advances
 * `*pos_io` past every token consumed or emitted. Writes decoded output
 * to stdout as tokens arrive. */
static turn_stats_t run_turn(bt_model_t *model,
                             bt_sampler_t *sampler,
                             const int *prompt_ids,
                             int n_prompt,
                             int *pos_io,
                             int *prev_token_io,
                             const chat_config_t *cfg) {
    turn_stats_t st = {0};
    bt_tokenizer_t *tok = &model->tokenizer;
    int pos = *pos_io;
    int prev_token = *prev_token_io;

    /* --- Prefill: run bt_forward on every prompt token in order. ---
     * Only the last prompt token's logits are sampled, so intermediate
     * tokens use bt_forward_prefill to skip the final norm + LM head. */
    double t0 = now_sec();
    for (int i = 0; i < n_prompt && !g_interrupted; i++) {
        if (pos >= model->config.max_seq_len) {
            fprintf(stderr,
                    "\n[chat] context full (pos=%d, max_seq_len=%d). "
                    "Use /reset to start a new conversation.\n",
                    pos, model->config.max_seq_len);
            *pos_io = pos;
            *prev_token_io = prev_token;
            return st;
        }
        if (i == n_prompt - 1) {
            bt_forward(model, prompt_ids[i], pos);
        } else {
            bt_forward_prefill(model, prompt_ids[i], pos);
        }
        prev_token = prompt_ids[i];
        pos++;
        st.prefill_tokens++;
    }
    st.prefill_sec = now_sec() - t0;

    /* --- Decode loop ------------------------------------------------ */
    int repeat_token = -1, repeat_count = 0;
    int repeat_limit = cfg->repeat_limit > 0 ? cfg->repeat_limit : 8;

    double t1 = now_sec();
    for (int i = 0; i < cfg->max_tokens && !g_interrupted; i++) {
        int next = bt_sample(sampler, model->state.logits,
                             model->config.vocab_size);

        /* End-of-turn / end-of-text — stop silently */
        if (next == tok->eos_id || next == tok->eot_id) break;

        /* User-configured stop tokens */
        if (cfg->n_stop_tokens > 0) {
            bool stop = false;
            for (int j = 0; j < cfg->n_stop_tokens; j++)
                if (next == cfg->stop_tokens[j]) { stop = true; break; }
            if (stop) break;
        }

        /* Degenerate-loop stop: same token N times in a row */
        if (next == repeat_token) {
            if (++repeat_count >= repeat_limit) break;
        } else {
            repeat_token = next;
            repeat_count = 1;
        }

        /* Stream decoded piece to stdout */
        const char *piece = bt_decode(tok, prev_token, next);
        fputs(piece, stdout);
        fflush(stdout);

        if (pos >= model->config.max_seq_len) {
            fprintf(stderr,
                    "\n[chat] context full (pos=%d). "
                    "Use /reset to continue.\n", pos);
            prev_token = next;
            pos++;
            st.generated_tokens++;
            break;
        }

        /* Forward the sampled token so the next step sees updated state */
        bt_forward(model, next, pos);
        prev_token = next;
        pos++;
        st.generated_tokens++;
    }
    st.decode_sec = now_sec() - t1;

    *pos_io = pos;
    *prev_token_io = prev_token;
    return st;
}

/* ---- Commands -------------------------------------------------------- */

static void print_help(void) {
    fprintf(stderr,
            "[chat] commands:\n"
            "  /reset  - clear conversation (rewind pos to 0)\n"
            "  /quit   - exit chat (alias: /exit, Ctrl-D)\n"
            "  /help   - show this help\n");
}

/* Handle a line starting with '/'. Returns:
 *    1 — command consumed, continue loop
 *   -1 — quit requested
 *    0 — not a recognized command (caller should treat as input) */
static int handle_command(const char *line, int *pos_io, int *prev_token_io,
                          bool *first_turn_io) {
    if (!strcmp(line, "/quit") || !strcmp(line, "/exit")) return -1;
    if (!strcmp(line, "/reset")) {
        *pos_io = 0;
        *prev_token_io = 0;
        *first_turn_io = true;
        fprintf(stderr, "[chat] conversation reset\n");
        return 1;
    }
    if (!strcmp(line, "/help")) { print_help(); return 1; }
    return 0;
}

/* ---- Public entry ---------------------------------------------------- */

int chat_run(bt_model_t *model, const chat_config_t *cfg) {
    struct sigaction sa = {0};
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT, &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    bt_sampler_t sampler;
    bt_sampler_init(&sampler, cfg->temperature, cfg->top_p, cfg->seed);

    int *prompt_ids = (int *)malloc(MAX_PROMPT_TOKENS * sizeof(int));
    if (!prompt_ids) {
        fprintf(stderr, "[chat] out of memory\n");
        return 1;
    }

    int pos = 0;
    int prev_token = 0;
    bool first_turn = true;
    bool is_tty = isatty(STDIN_FILENO);

    /* Pipe mode (or non-tty stdin): read one prompt, answer, exit. */
    if (cfg->pipe_mode || !is_tty) {
        char *input = read_stdin_all();
        if (!input) {
            fprintf(stderr, "[chat] empty input\n");
            free(prompt_ids);
            return 1;
        }

        char *prompt = apply_template(input, cfg->prompt_tpl);
        free(input);

        int n = bt_encode(&model->tokenizer, prompt, prompt_ids,
                          MAX_PROMPT_TOKENS, /*add_bos=*/1);
        free(prompt);
        if (n <= 0) {
            fprintf(stderr, "[chat] tokenizer produced 0 tokens\n");
            free(prompt_ids);
            return 1;
        }
        if (n >= MAX_PROMPT_TOKENS)
            fprintf(stderr, "[chat] warning: prompt truncated at %d tokens\n",
                    MAX_PROMPT_TOKENS);

        turn_stats_t st = run_turn(model, &sampler, prompt_ids, n,
                                   &pos, &prev_token, cfg);
        fputc('\n', stdout);
        fflush(stdout);
        if (st.generated_tokens > 0 && st.decode_sec > 0.0) {
            double tps = (double)st.generated_tokens / st.decode_sec;
            fprintf(stderr,
                    "[chat] %d tok, %.2f tok/s | prefill %d tok in %.2fs\n",
                    st.generated_tokens, tps, st.prefill_tokens,
                    st.prefill_sec);
        }
        free(prompt_ids);
        return 0;
    }

    /* Interactive REPL */
    fprintf(stderr,
            "[chat] biturbo assistant ready (%d layers, dim=%d, "
            "max_seq_len=%d)\n"
            "[chat] type /help for commands, /quit to exit\n",
            model->config.n_layers, model->config.dim,
            model->config.max_seq_len);

    while (!g_interrupted) {
        char *line = read_line_tty("> ");
        if (!line) break; /* EOF / Ctrl-D */

        if (line[0] == '\0') { free(line); continue; }

        if (line[0] == '/') {
            int r = handle_command(line, &pos, &prev_token, &first_turn);
            free(line);
            if (r == -1) break;
            if (r == 1) continue;
            fprintf(stderr, "[chat] unknown command (try /help)\n");
            continue;
        }

        g_interrupted = 0;

        char *prompt = apply_template(line, cfg->prompt_tpl);
        free(line);

        /* Only the very first turn gets a BOS — subsequent turns continue
         * the existing conversation at the current KV cache position. */
        int n = bt_encode(&model->tokenizer, prompt, prompt_ids,
                          MAX_PROMPT_TOKENS,
                          /*add_bos=*/first_turn ? 1 : 0);
        free(prompt);
        if (n <= 0) {
            fprintf(stderr, "[chat] (no tokens)\n");
            continue;
        }
        if (n >= MAX_PROMPT_TOKENS)
            fprintf(stderr, "[chat] warning: prompt truncated at %d tokens\n",
                    MAX_PROMPT_TOKENS);

        turn_stats_t st = run_turn(model, &sampler, prompt_ids, n,
                                   &pos, &prev_token, cfg);
        fputc('\n', stdout);
        fflush(stdout);
        first_turn = false;

        if (st.generated_tokens > 0 && st.decode_sec > 0.0) {
            double tps = (double)st.generated_tokens / st.decode_sec;
            fprintf(stderr,
                    "[chat] %d tok, %.2f tok/s | prefill %d tok in %.2fs "
                    "| pos=%d/%d\n",
                    st.generated_tokens, tps, st.prefill_tokens,
                    st.prefill_sec, pos, model->config.max_seq_len);
        } else {
            fprintf(stderr, "[chat] (no output) | pos=%d/%d\n",
                    pos, model->config.max_seq_len);
        }

        g_interrupted = 0;
    }

    fputc('\n', stderr);
    free(prompt_ids);
    return 0;
}
