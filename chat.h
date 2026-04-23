/*
 * biturbo chat mode — interactive multi-turn assistant
 *
 * Unlike BitMamba (fixed-size SSM state), biturbo uses a growing KV cache
 * keyed by position. Multi-turn chat continues to grow `pos` across turns
 * until the cache is full (config.max_seq_len); `/reset` rewinds pos to 0
 * so the next turn overwrites stale entries.
 *
 * State save/load is intentionally omitted: the KV cache is huge and
 * model-version-specific — reloading a conversation would cost more than
 * just replaying the prefill.
 */

#ifndef BITURBO_CHAT_H
#define BITURBO_CHAT_H

#include "biturbo.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CHAT_MAX_STOP_TOKENS 8

typedef struct {
    float temperature;
    float top_p;
    uint64_t seed;
    int max_tokens;                        /* per turn */
    const char *prompt_tpl;                /* "%s" placeholder, NULL = raw */
    bool pipe_mode;                        /* true = single stdin prompt, no REPL */
    int repeat_limit;                      /* same-token run length (0 = default) */
    int stop_tokens[CHAT_MAX_STOP_TOKENS]; /* extra stop token IDs */
    int n_stop_tokens;
} chat_config_t;

/* Rewind conversation so the next turn starts at pos=0. Old KV entries
 * remain in memory but are overwritten as new tokens are fed in. */
void chat_reset(bt_model_t *model);

/* Run REPL (tty) or single-shot (pipe_mode / non-tty) chat.
 * Returns 0 on clean exit, 1 on error. */
int chat_run(bt_model_t *model, const chat_config_t *cfg);

#ifdef __cplusplus
}
#endif

#endif /* BITURBO_CHAT_H */
