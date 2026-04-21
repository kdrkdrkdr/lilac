# Minimal dev Makefile for lilac C engine. Expects gcc (MinGW/MSYS2).
# OpenBLAS + miniaudio are headered here; link them as each target needs them.
#
# GNU make sets CC=cc by default (implicitly), so ?= won't override it.
# Use = to force gcc unless the user explicitly overrides on the command line.

CC        = gcc
CFLAGS   ?= -O2 -Wall -Wextra -std=c11 -mavx2 -mfma
OBLAS_INC = -Iopenblas/include
OBLAS_LIB = -Lopenblas/lib -lopenblas

# Static link for libvc.dll so we don't have to ship libopenblas.dll alongside.
# Force .a over .dll.a, then swap back to dynamic for system libs.
OBLAS_LIB_STATIC = -Wl,-Bstatic -Lopenblas/lib -lopenblas -lgfortran -lquadmath -lpthread -Wl,-Bdynamic

# RNNoise (bundled under cpp/rnnoise/). RTCD picks AVX2 at runtime; compiling
# nnet_sse4_1/nnet_avx2 with our global -mavx2 is fine (they're gated by
# CPU detection). CPU_INFO_BY_C uses gcc's <cpuid.h>. FLOAT_APPROX is the
# upstream default for rnnoise/libopus style code.
RNNOISE_INC  = -Irnnoise -Irnnoise/src
RNNOISE_DEFS = -DRNN_ENABLE_X86_RTCD -DCPU_INFO_BY_C -DFLOAT_APPROX
RNNOISE_SRCS = rnnoise/src/denoise.c rnnoise/src/rnn.c rnnoise/src/pitch.c \
               rnnoise/src/kiss_fft.c rnnoise/src/celt_lpc.c rnnoise/src/nnet.c \
               rnnoise/src/nnet_default.c rnnoise/src/parse_lpcnet_weights.c \
               rnnoise/src/rnnoise_data.c rnnoise/src/rnnoise_tables.c \
               rnnoise/src/x86/nnet_sse4_1.c rnnoise/src/x86/nnet_avx2.c \
               rnnoise/src/x86/x86_dnn_map.c rnnoise/src/x86/x86cpu.c

ENGINE_SRCS = log.c pool.c tensor.c ops.c conv.c stream.c gru.c stft.c wn.c ref_enc.c enc_q.c flow.c dec.c dec_stream.c model.c engine.c audio_io.c $(RNNOISE_SRCS)
ENGINE_HDRS = log.h pool.h tensor.h ops.h conv.h stream.h gru.h stft.h wn.h ref_enc.h enc_q.h flow.h dec.h dec_stream.h model.h engine.h audio_io.h miniaudio.h

ENGINE_INC  = $(OBLAS_INC) $(RNNOISE_INC)
ENGINE_DEFS = $(RNNOISE_DEFS)

all: test_tensor.exe test_ops.exe test_conv.exe test_extras.exe test_engine.exe libvc.dll

test_engine.exe: test_engine.c $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_engine.c $(ENGINE_SRCS) $(OBLAS_LIB) -lole32 -lwinmm -lksuser

bench.exe: bench.c $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ bench.c $(ENGINE_SRCS) $(OBLAS_LIB) -lole32 -lwinmm -lksuser

test_model.exe: test_model.c $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_model.c $(ENGINE_SRCS) $(OBLAS_LIB) -lole32 -lwinmm -lksuser

test_stream.exe: test_stream.c $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_stream.c $(ENGINE_SRCS) $(OBLAS_LIB) -lole32 -lwinmm -lksuser


test_tensor.exe: test_tensor.c tensor.c tensor.h
	$(CC) $(CFLAGS) -o $@ test_tensor.c tensor.c

test_ops.exe: test_ops.c ops.c ops.h
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_ops.c ops.c $(OBLAS_LIB)

test_conv.exe: test_conv.c conv.c conv.h ops.c ops.h tensor.c tensor.h
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_conv.c conv.c ops.c tensor.c $(OBLAS_LIB)

test_extras.exe: test_extras.c conv.c conv.h gru.c gru.h ops.c ops.h tensor.c tensor.h
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_extras.c conv.c gru.c ops.c tensor.c $(OBLAS_LIB)

test_simd_math.exe: test_simd_math.c simd_math.h
	$(CC) $(CFLAGS) -o $@ test_simd_math.c

test_conv_stream.exe: test_conv_stream.c conv.c conv.h stream.c stream.h ops.c ops.h simd_math.h
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_conv_stream.c conv.c stream.c ops.c $(OBLAS_LIB)

test_dec_stream.exe: test_dec_stream.c $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_INC) $(ENGINE_DEFS) -o $@ test_dec_stream.c $(ENGINE_SRCS) $(OBLAS_LIB) -lole32 -lwinmm -lksuser

libvc.dll: vc.c vc.h $(ENGINE_SRCS) $(ENGINE_HDRS)
	$(CC) $(CFLAGS) $(ENGINE_DEFS) -DLILAC_BUILD_DLL -shared $(ENGINE_INC) -o $@ vc.c $(ENGINE_SRCS) $(OBLAS_LIB_STATIC) -static-libgcc -lole32 -lwinmm -lksuser

clean:
	rm -f *.exe *.o *.dll
