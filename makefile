BUILD_PATH = build
CL_PATH = cl

LIB = -L C:\Users\Nathan\vcpkg\installed\x64-windows\lib -lOpencl -L lib
INC = -I C:\Users\Nathan\vcpkg\installed\x64-windows\include

SRC = $(wildcard *.cpp)
CLSRC = $(wildcard *.cl)
CLCPP = $(patsubst %.cl,$(CL_PATH)/%.cpp,$(CLSRC))
OBJ = $(patsubst %.cpp,$(BUILD_PATH)/%.o,$(SRC))

lattice: $(OBJ)
	$(info $$SRC is [${SRC}])
	$(info $$OBJ is [${OBJ}])
	$(info $$^ is [$^])
	g++ -Wall -o $@ $^ $(LIB) $(INC)
	
$(BUILD_PATH)/clsrc.o: $(BUILD_PATH)/clsrc.cpp
	g++ -Wall -c -o $@ $^ $(LIB) $(INC)
	
$(BUILD_PATH)/%.o: %.cpp | $(CLCPP) $(BUILD_PATH)
	g++ -Wall -c -o $@ $< $(LIB) $(INC)

$(CL_PATH)/%.cpp: %.cl | $(CL_PATH)
	@echo Writing $< to $@
	@echo "const char $*[] = {" > $@
	@hexdump -v -e '1/1 "0x%02x, "' $^ >> $@
	@echo "0x00};" >> $@
	
$(BUILD_PATH): 
	@mkdir $(BUILD_PATH)
	
$(CL_PATH): 
	@mkdir $(CL_PATH)
	
.PHONY: clean
clean:
	rm -rf build
	rm -rf cl