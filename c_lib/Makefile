CC = gcc
# Default flags
CFLAGS = -Wall -Wextra -fPIC
LDFLAGS = -shared

LIB_NAME = libGeoLDPC.so
EXE_NAME = test_decoder
SRCS = crc24a.c \
	   ldpc_tables.c \
	   int8_ldpc_decoder.c \
	   int16_ldpc_decoder.c \

# Default build (Release)
all: CFLAGS += -O3
all: $(LIB_NAME) $(EXE_NAME)

# Debug build
debug: CFLAGS += -g -O0 -DDEBUG
debug: $(LIB_NAME) $(EXE_NAME)

$(LIB_NAME): $(SRCS:.c=.o)
	$(CC) $(LDFLAGS) -o $@ $^


$(EXE_NAME): test_ldpc_decoder.c $(SRCS:.c=.o)
	$(CC) $(CFLAGS) -o $@ $^

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f *.o $(LIB_NAME) $(EXE_NAME)

.PHONY: all debug clean
