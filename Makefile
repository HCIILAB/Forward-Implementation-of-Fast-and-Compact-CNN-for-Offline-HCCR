# This Makefile requires opencv and MKL in the compilation

CC=g++
MKDIR=mkdir -p
RM=rm -rf
SRC_DIR=src
OBJ_DIR=obj
BIN_DIR=bin
SRCS=$(wildcard $(SRC_DIR)/*.cpp)
OBJS=$(SRCS:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
CPPFLAGS=-I/usr/include/opencv -I/opt/intel/mkl/include
CFLAGS= -std=c++11 -O3
LDFLAGS= -L /opt/intel/mkl/lib/intel64
LDLIBS= -lopencv_core -lopencv_highgui -lopencv_imgproc -lmkl_rt 
TARGET=$(BIN_DIR)/run

.PHONY: all run clean

all: $(BIN_DIR) $(OBJ_DIR) $(TARGET)

$(BIN_DIR):
	$(MKDIR) $(BIN_DIR)
$(OBJ_DIR):
	$(MKDIR) $(OBJ_DIR)
$(TARGET): $(OBJS)
	$(CC) $(LDFLAGS) $^ $(LDLIBS) -o $@
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

run:
	$(TARGET)

clean:
	$(RM) $(BIN_DIR) $(OBJ_DIR) $(OBJS) $(TARGET)