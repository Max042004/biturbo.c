CC      = cc
CFLAGS  = -std=gnu99 -O2 -Wall -Wextra -Wpedantic -DNDEBUG
LDFLAGS = -lm

# ARM NEON (auto-detected on macOS Apple Silicon)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  CFLAGS += -mcpu=apple-m1
endif
ifeq ($(UNAME_M),aarch64)
  CFLAGS += -march=armv8-a+fp+simd
endif

TARGET     = biturbo
PACKER     = pack_btpk
SRCS       = main.c biturbo.c
OBJS       = $(SRCS:.c=.o)
PACK_OBJS  = pack_btpk.o biturbo.o

.PHONY: all clean debug fpga pack cma-module cma-clean

all: $(TARGET) $(PACKER)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(PACKER): $(PACK_OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c biturbo.h biturbo_btpk.h
	$(CC) $(CFLAGS) -c -o $@ $<

debug: CFLAGS = -std=gnu99 -O0 -g -Wall -Wextra -Wpedantic -fsanitize=address,undefined
debug: LDFLAGS += -fsanitize=address,undefined
debug: clean $(TARGET) $(PACKER)

# FPGA build: cross-compile for DE10-Nano ARM with T-MAC accelerator
FPGA_CC     ?= arm-linux-gnueabihf-gcc
FPGA_CFLAGS  = -std=gnu99 -O3 -Wall -Wextra -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -funsafe-math-optimizations -fopenmp -DBT_FPGA -DNDEBUG
FPGA_LDFLAGS = -lm -fopenmp
KDIR        ?= /lib/modules/$(shell uname -r)/build
# Linux 4.14 arm hard-float toolchains can mis-detect ARMv7 during cc-option
# probing and fall back to -march=armv5t unless we force the module build flags.
CMA_KCFLAGS ?= -march=armv7-a -Wa,-march=armv7-a -mfpu=vfp

fpga:
	$(FPGA_CC) $(FPGA_CFLAGS) -o biturbo_fpga main.c biturbo.c $(FPGA_LDFLAGS)

cma-module:
	$(MAKE) -C $(KDIR) M=$(CURDIR)/kernel KCFLAGS="$(CMA_KCFLAGS)" modules

cma-clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR)/kernel KCFLAGS="$(CMA_KCFLAGS)" clean

clean:
	rm -f $(OBJS) $(PACK_OBJS) $(TARGET) $(PACKER) biturbo_fpga
