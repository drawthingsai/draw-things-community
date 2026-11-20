#ifndef LOCAL_CODE_OPENSSH_PORT_H
#define LOCAL_CODE_OPENSSH_PORT_H

// ios_error.h redirects stdio/process calls into ios_system. Import the wide
// character declarations first so its compatibility macros do not rewrite
// declarations in the Apple SDK headers.
#include <wchar.h>
#include <wctype.h>

#include "ios_error.h"

// Sessions run commands concurrently in one process, so OpenSSH parses its
// options with the invocation's getopt (ios_system_support), not libc's
// process-wide state. The globals become lvalues of this thread's state, the
// way errno is, so OpenSSH's own `extern int optind;` lines stay valid.
#include <getopt.h>
#include <unistd.h>
int ios_system_getopt(int argc, char *const *argv, const char *options);
int *ios_system_optind_location(void);
int *ios_system_opterr_location(void);
int *ios_system_optopt_location(void);
int *ios_system_optreset_location(void);
char **ios_system_optarg_location(void);
#define getopt ios_system_getopt
#define optind (*ios_system_optind_location())
#define opterr (*ios_system_opterr_location())
#define optopt (*ios_system_optopt_location())
#define optreset (*ios_system_optreset_location())
#define optarg (*ios_system_optarg_location())

// The thread's own character type, for msetlocale (see utf8.c).
#include <xlocale.h>
locale_t ios_system_ctype_locale(const char *name);

// Session-relative paths (ios_system_support): includes.h maps the libc path
// calls to these once the system headers are in. Including those headers here
// keeps a later include from declaring the mapped names.
#include <dirent.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/time.h>
void ios_system_support_init_directory(void);
int ios_system_open(const char *path, int flags, ...);
FILE *ios_system_fopen(const char *path, const char *mode);
int ios_system_stat(const char *path, struct stat *info);
int ios_system_lstat(const char *path, struct stat *info);
int ios_system_access(const char *path, int mode);
int ios_system_unlink(const char *path);
int ios_system_rename(const char *source, const char *destination);
int ios_system_link(const char *source, const char *destination);
int ios_system_symlink(const char *target, const char *path);
ssize_t ios_system_readlink(const char *path, char *destination, size_t size);
int ios_system_mkdir(const char *path, mode_t mode);
int ios_system_rmdir(const char *path);
int ios_system_chmod(const char *path, mode_t mode);
int ios_system_utimes(const char *path, const struct timeval times[2]);
DIR *ios_system_opendir(const char *path);
char *ios_system_realpath(const char *path, char *resolved);
int ios_system_mkstemp(char *path);
char *ios_system_mkdtemp(char *path);
int ios_system_chdir(const char *path);
char *ios_system_getcwd(char *buffer, size_t size);

#endif
