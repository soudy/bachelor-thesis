.PHONY: all

all: $(wildcard *.tex)
	latexmk -pdf -use-make $^

clean:
	latexmk -CA
	@rm -f graduate-thesis.bbl
