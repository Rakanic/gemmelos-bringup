PREFIX = riscv64-unknown-elf-

CC = $(PREFIX)gcc
CXX = $(PREFIX)g++
CP = $(PREFIX)objcopy
OD = $(PREFIX)objdump
DG = $(PREFIX)gdb
SIZE = $(PREFIX)size

PORT = 3333
TYPE = Release

.PHONY: build
build:
	cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=$(TYPE) -D CMAKE_TOOLCHAIN_FILE=./riscv-gcc.cmake -DCHIP=$(CHIP) $(if $(PLATFORM), -D PLATFORM=$(PLATFORM),) $(if $(VECNN), -D BUILD_VECNN=$(VECNN),) $(if $(RVV), -D ENABLE_RVV=$(RVV),) $(if $(RVV_TYPE), -D RVV_TYPE=$(RVV_TYPE),) $(if $(THREAD_LIB), -D THREAD_LIB=$(THREAD_LIB),) $(if $(BMARK_LIB), -D BMARK_LIB=$(BMARK_LIB),) $(EXTRA_CMAKE_ARGS)
	cmake --build ./build/ --target $(TARGET)

.PHONY: ocd
ocd:
	openocd -f ./platform/$(CHIP)/$(CHIP).cfg

.PHONY: ocd-run
ocd-run:
	openocd -f ./platform/$(CHIP)/$(CHIP).cfg -c "reset run" -c "halt" -c "load_image $(BINARY)" -c "resume 0x80000000"

.PHONY: gdb
gdb:
	$(DG) $(BINARY) --eval-command="target extended-remote localhost:$(PORT)"
	# --eval-command="monitor reset"

.PHONY: clean
clean:
	rm -rf build

.PHONY: dump
dump:
	$(OD) -D  $(BINARY) > $(BINARY).dump

.PHONY: checktsi
checktsi:
	uart_tsi +tty=$(TTY) +baudrate=921600 +no_hart0_msip +init_write=0x80001000:0xb0bacafe +init_read=0x80001000 none

.PHONY: tsi-run
tsi-run:
	uart_tsi +tty=$(TTY) +baudrate=921600 $(BINARY)

# Run a scratchpad-linked binary (built with LINKER=scratch, i.e. everything at 0x08000000).
# The bootrom jumps to whatever address the boot-address register holds (default 0x80000000 =
# DRAM), so we reprogram it first. uart_tsi applies +init_write AFTER loading the program and
# just BEFORE releasing hart 0's MSIP, so the new vector is in place when the core starts.
# BOOTADDR_REG lives at 0x1000 (PBUS); it is XLen-wide but its upper word is already 0, and
# +init_write is a 32-bit store, so writing the low word is enough.
BOOTADDR_REG ?= 0x1000
BOOTADDR     ?= 0x08000000

.PHONY: tsi-run-spad
tsi-run-spad:
	uart_tsi +tty=$(TTY) +baudrate=921600 +init_write=$(BOOTADDR_REG):$(BOOTADDR) $(BINARY)

.PHONY: vcs-run
vcs-run:
	echo "Running VCS tests within $(CY_DIR)"
	(cd $(CY_DIR)/sims/vcs && make run-binary CONFIG=$(CONFIG) LOADMEM=1 BINARY=$(BINARY))

BINARY_REL := $(patsubst $(realpath ../../sims/vcs)/%,%,$(abspath $(BINARY)))

.PHONY: run
run:
	srun -p ee194 --pty make -C ../../sims/vcs run-binary LOADMEM=1 CONFIG=$(CONFIG) BINARY=$(BINARY_REL)
# 	(cd $(CY_DIR)/sims/vcs && make run-binary-debug-hex CONFIG=$(CONFIG) LOADMEM=1 BINARY=$(BINARY))
