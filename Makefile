PLATFORM ?= PLATFORM_DESKTOP

RAYLIB_DIR = C:/raylib
INCLUDE_DIR = -I ./ -I $(RAYLIB_DIR)/raylib/src -I $(RAYLIB_DIR)/raygui/src
LIBRARY_DIR = -L $(RAYLIB_DIR)/raylib/src
DEFINES =  -D _DEFAULT_SOURCE -D RAYLIB_BUILD_MODE=RELEASE -D $(PLATFORM)

ifeq ($(PLATFORM),PLATFORM_DESKTOP)
    CC = g++
    EXT = .exe
    CFLAGS ?= $(DEFINES) -O3 $(RAYLIB_DIR)/raylib/src/raylib.rc.data $(INCLUDE_DIR) $(LIBRARY_DIR) 
    LIBS = -lraylib -lopengl32 -lgdi32 -lwinmm
endif

ifeq ($(PLATFORM),PLATFORM_WEB)
    CC = emcc
    EXT = .html
    CFLAGS ?= $(DEFINES) $(RAYLIB_DIR)/raylib/src/libraylib.bc -Os -s USE_GLFW=3 -s FORCE_FILESYSTEM=1 -s MAX_WEBGL_VERSION=2 -s ALLOW_MEMORY_GROWTH=1 --preload-file $(dir $<)resources@resources --shell-file ./shell.html $(INCLUDE_DIR) $(LIBRARY_DIR)
endif

SOURCE = $(wildcard *.cpp)
HEADER = $(wildcard *.h)

.PHONY: all

all: looping

looping: $(SOURCE) $(HEADER)
	$(CC) -o $@$(EXT) $(SOURCE) $(CFLAGS) $(LIBS) 

clean:
	rm looping$(EXT)