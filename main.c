/*
 * main.c — biturbo CLI runner
 *
 * Usage: biturbo <model.gguf> [options]
 *   -p <prompt>      Input prompt (default: "Hello")
 *   -n <count>       Max tokens to generate (default: 256)
 *   -t <temp>        Temperature (default: 0.8, 0.0 = greedy)
 *   -k <top_p>       Top-p nucleus sampling (default: 0.9)
 *   -s <seed>        RNG seed (default: time-based)
 *   --chat           Enter interactive chat mode
 *   --template <fmt> Chat prompt template (%%s = user input)
 *   --no-template    Feed raw user input to the model
 *   --stop-token ID  Extra stop token (may be repeated)
 *   --repeat-limit N Stop after N identical tokens in a row
 */

#include "biturbo.h"
#include "chat.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_CHAT_TEMPLATE "User: %s\nAssistant:"

static void usage(const char* prog) {
    fprintf(stderr,
        "biturbo — BitNet 1.58-bit inference with TurboQuant INT4 KV cache\n\n"
        "Usage: %s <model.gguf> [options]\n\n"
        "Options:\n"
        "  -p <prompt>      Input prompt (default: \"Hello\")\n"
        "  -n <count>       Max tokens to generate (default: 256)\n"
        "  -t <temp>        Temperature, 0.0 = greedy (default: 0.8)\n"
        "  -k <top_p>       Top-p nucleus sampling (default: 0.9)\n"
        "  -s <seed>        RNG seed (default: time-based)\n"
        "  --chat           Interactive chat (REPL; also accepts stdin pipe)\n"
        "  --template <fmt> Chat prompt template with %%s placeholder\n"
        "                   (default: \"User: %%s\\nAssistant:\")\n"
        "  --no-template    Pass raw user input to the model\n"
        "  --stop-token ID  Extra stop token (may be repeated)\n"
        "  --repeat-limit N Stop after N identical tokens in a row\n"
        "  -h               Show this help\n\n"
        "Loads GGUF models with I2_S (1.58-bit ternary) weights.\n"
        "KV cache quantized to INT4 via TurboQuant uniform scheme.\n\n"
        "Chat examples:\n"
        "  %s --chat model.gguf\n"
        "  echo \"Who are you?\" | %s --chat model.gguf\n",
        prog, prog, prog);
}

int main(int argc, char** argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    const char* model_path = NULL;
    const char* prompt = "Hello";
    int max_tokens = 256;
    float temperature = 0.8f;
    float top_p = 0.9f;
    uint64_t seed = 0;

    bool chat_mode = false;
    bool no_template = false;
    const char* chat_template = NULL;
    int repeat_limit = 0;
    int stop_tokens[CHAT_MAX_STOP_TOKENS];
    int n_stop_tokens = 0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') { model_path = argv[i]; continue; }
        if (strcmp(argv[i], "-h") == 0) { usage(argv[0]); return 0; }
        if (strcmp(argv[i], "--chat") == 0) { chat_mode = true; continue; }
        if (strcmp(argv[i], "--no-template") == 0) { no_template = true; continue; }
        if (i + 1 >= argc) {
            fprintf(stderr, "error: %s needs argument\n", argv[i]);
            return 1;
        }
        if (strcmp(argv[i], "--template") == 0) {
            chat_template = argv[++i];
            continue;
        }
        if (strcmp(argv[i], "--stop-token") == 0) {
            if (n_stop_tokens >= CHAT_MAX_STOP_TOKENS) {
                fprintf(stderr, "error: too many --stop-token (max %d)\n",
                        CHAT_MAX_STOP_TOKENS);
                return 1;
            }
            stop_tokens[n_stop_tokens++] = atoi(argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "--repeat-limit") == 0) {
            repeat_limit = atoi(argv[++i]);
            continue;
        }
        switch (argv[i][1]) {
            case 'p': prompt = argv[++i]; break;
            case 'n': max_tokens = atoi(argv[++i]); break;
            case 't': temperature = (float)atof(argv[++i]); break;
            case 'k': top_p = (float)atof(argv[++i]); break;
            case 's': seed = (uint64_t)atoll(argv[++i]); break;
            default:
                fprintf(stderr, "unknown option '%s'\n", argv[i]);
                return 1;
        }
    }

    if (!model_path) { usage(argv[0]); return 1; }

    bt_model_t model;
    if (bt_load_model(&model, model_path) != 0) return 1;

    bt_config_t* cfg = &model.config;
    fprintf(stderr, "biturbo: %d layers, %d/%d heads, head_dim=%d, "
            "KV cache TQ4 (RHT+codebook+QJL, %d blk/head)\n",
            cfg->n_layers, cfg->n_heads, cfg->n_kv_heads,
            BT_HEAD_DIM(cfg),
            (BT_HEAD_DIM(cfg) + BT_QK - 1) / BT_QK);

    int ret = 0;
    if (chat_mode) {
        chat_config_t ccfg = {
            .temperature  = temperature,
            .top_p        = top_p,
            .seed         = seed,
            .max_tokens   = max_tokens,
            .prompt_tpl   = no_template ? NULL
                          : (chat_template ? chat_template
                                           : DEFAULT_CHAT_TEMPLATE),
            .pipe_mode    = false,
            .repeat_limit = repeat_limit,
            .n_stop_tokens = n_stop_tokens,
        };
        for (int i = 0; i < n_stop_tokens; i++)
            ccfg.stop_tokens[i] = stop_tokens[i];
        ret = chat_run(&model, &ccfg);
    } else {
        bt_sampler_t sampler;
        bt_sampler_init(&sampler, temperature, top_p, seed);
        bt_generate(&model, &sampler, prompt, max_tokens);
    }

    bt_free_model(&model);
    return ret;
}
