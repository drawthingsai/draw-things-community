#ifndef VENDORS_OSH_OSH_IOS_H
#define VENDORS_OSH_OSH_IOS_H

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
  OSH_IOS_STATUS_UNSUPPORTED = 125,
  OSH_IOS_STATUS_CANCELLED = 130,
};

typedef int (*osh_ios_external_runner)(
    int argc, const char* const argv[], int envc,
    const char* const environment[], FILE* input, FILE* output, FILE* error,
    void* context);

typedef int (*osh_ios_is_cancelled)(void* context);
typedef void (*osh_ios_signal_handler)(int signal_number);
typedef int (*osh_ios_send_signal)(int identifier, int signal_number,
                                    int process_group, void* context);
typedef int (*osh_ios_set_signal_handler)(int signal_number,
                                           osh_ios_signal_handler handler,
                                           void* context);
typedef int (*osh_ios_get_process_id)(void* context);
/* Return 1 for an executable virtual path, 0 for a missing virtual path,
 * or -1 to let OSH inspect the real filesystem. */
typedef int (*osh_ios_lookup_virtual_path)(const char* path, void* context);

/* The runner owns each distinct stream until that stage exits, closes it then,
 * and fills every status before returning. Arguments remain valid for the call.
 * Pipeline stages must run concurrently; no process-wide stdio is redirected. */
typedef struct osh_ios_pipeline_command {
  int argc;
  const char* const* argv;
  FILE* input;
  FILE* output;
  FILE* error;
  /* If non-null, write this diagnostic to error and return status 1 instead of
   * running the command. Close the stage streams normally in either case. */
  const char* redirect_error;
} osh_ios_pipeline_command;
typedef void (*osh_ios_pipeline_runner)(
    int count, const osh_ios_pipeline_command commands[], int envc,
    const char* const environment[], int statuses[], void* context);

typedef struct osh_ios_config {
  osh_ios_external_runner run_external;
  osh_ios_is_cancelled is_cancelled;
  osh_ios_send_signal send_signal;
  osh_ios_set_signal_handler set_signal_handler;
  osh_ios_get_process_id get_process_id;
  osh_ios_lookup_virtual_path lookup_virtual_path;
  int environment_count;
  const char* const* environment;
  void* context;
  osh_ios_pipeline_runner run_pipeline;
  /* Optional capability check, called before opening redirects or pipe streams. */
  int (*can_run_pipeline)(int count, const char* const names[], void* context);
} osh_ios_config;

int osh_ios_run(const char* command, FILE* input, FILE* output, FILE* error,
                 const osh_ios_config* config);

#ifdef __cplusplus
}

namespace osh_ios {

const osh_ios_config* CurrentConfig();
bool IsCancelled();
void MarkUnsupported();
bool WasUnsupported();

}  // namespace osh_ios
#endif

#endif  // VENDORS_OSH_OSH_IOS_H
