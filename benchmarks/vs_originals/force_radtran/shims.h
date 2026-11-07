/* What FORCE's own functions need around them (copied verbatim into radtran_harness.c by make_harness.sh): the headers
 * of the files that are compiled as they are. FORCE 3.11.00, Copyright (C) 2013-2022 David Frantz, GPL-3.0-or-later;
 * this file is Zeit's, same license. */
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "../cross-level/const-cl.h"
#include "../cross-level/date-cl.h"
#include "../cross-level/sun-cl.h"
#include "../lower-level/radtran-ll.h"
#include "../lower-level/table-ll.h"
