test:
	py.test -v --cov=autoda
	make test-notebooks

tests: test

test-notebooks:
	for notebook in 'find ../notebooks/* -name ".ipynb"'; do \
	    py.test --nbval-lax $$notebook ; \
	done



doc:
	make -C ../docs html 


clean:
	rm -rf *.pyc __pycache__
	rm -rf */*.pyc */__pycache__
	rm -rf */*/*.pyc */__pycache__

all: test doc 
.PHONY: all
