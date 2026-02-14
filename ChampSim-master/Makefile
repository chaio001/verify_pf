app = champsim

srcExt = cc
srcDir = src branch replacement prefetcher
objDir = obj
binDir = bin
inc = inc
inc = inc HyperStream/include

debug = 1

CFlags = -Wall -O3 -std=c++17
LDFlags =
libs =
libDir =


PREFETCHER ?= scooby

$(info PREFETCHER = $(PREFETCHER))



#************************ DO NOT EDIT BELOW THIS LINE! ************************

ifeq ($(debug),1)
	debug=-g
else
	debug=
endif
inc := $(addprefix -I,$(inc))
libs := $(addprefix -l,$(libs))
libDir := $(addprefix -L,$(libDir))
CFlags += -c $(debug) $(inc) $(libDir) $(libs)
########################################################################################
# ================= Source Selection =================

ALL_SOURCES := $(shell find $(srcDir) -name '*.$(srcExt)')

# -------- Default: Scooby --------
# ifeq ($(PREFETCHER),from_file)
ifneq (,$(filter $(PREFETCHER),from_file no bo sisb sisb_bo pathfinder))
EXCLUDE_SOURCES := \
	prefetcher/hdc_pf.cc \
	prefetcher/hdc_pf_helper.cc \
	src/learning_engine_hdc.cc \
	src/learning_engine_hdc_basic.cc \
	src/learning_engine_hdc_featurewise.cc \
	src/hdc_feature_knowledge.cc \
	src/hdc_util.cc

endif

# -------- HDC-PF --------
ifeq ($(PREFETCHER),hdc)
EXCLUDE_SOURCES := \
	prefetcher/scooby.cc \
	prefetcher/scooby_helper.cc \
	src/learning_engine_base.cc \
	src/learning_engine_basic.cc \
	src/learning_engine_featurewise.cc \
	src/feature_knowledge.cc \
	src/util.cc
endif

sources := $(filter-out $(EXCLUDE_SOURCES), $(ALL_SOURCES))

srcDirs := $(shell find . -name '*.$(srcExt)' -exec dirname {} \; | uniq)
objects := $(patsubst %.$(srcExt),$(objDir)/%.o,$(sources))
########################################################################################
# sources := $(shell find $(srcDir) -name '*.$(srcExt)')
# srcDirs := $(shell find . -name '*.$(srcExt)' -exec dirname {} \; | uniq)
# objects := $(patsubst %.$(srcExt),$(objDir)/%.o,$(sources))
########################################################################################


ifeq ($(srcExt),cc)
	CC = $(CXX)
else
	CFlags += -std=gnu99
endif

.phony: all clean distclean


all: $(binDir)/$(app)

$(binDir)/$(app): buildrepo $(objects)
	@mkdir -p `dirname $@`
	@echo "Linking $@..."
	@$(CC) $(objects) $(LDFlags) -o $@

$(objDir)/%.o: %.$(srcExt)
	@echo "Generating dependencies for $<..."
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	@echo "Compiling $<..."
	@$(CC) $(CFlags) $< -o $@

clean:
	$(RM) -r $(objDir)

distclean: clean
	$(RM) -r $(binDir)/$(app)

buildrepo:
	@$(call make-repo)

define make-repo
   for dir in $(srcDirs); \
   do \
	mkdir -p $(objDir)/$$dir; \
   done
endef


# usage: $(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(CC) -MM       \
        -MF $3    \
        -MP       \
        -MT $2    \
        $(CFlags) \
        $1
endef
