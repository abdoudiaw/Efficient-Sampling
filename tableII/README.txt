Adjustable settings are in _model.py, but are set at values used in manuscript.
Results may be slightly different due to randomness from sampler and optimizer,
and the code has been slightly refactored over time.

Executes with:
  $ python main_workflow.py
and may take anywhere from a few minutes to a few hours, depending on the model
and tolerance used, and randomness.
Relevant information is printed to stdout, and also dumped into several files:
- cost.pkl: the objective function
- hist.pkl: historical testing misfit
- size.pkl: historical endpoint cache size
- score.txt: convergence of the test score versus model evaluations
- stop: a database of the endpoints of the optimizers
- func.db: a database of the learned surrogates
- eval: a database of evaluations of the objective

Any of the "pkl" files can be read like this:
  >>> import dill; cost = dill.load(open('cost.pkl', 'rb'))
while relevant results from the databases are plotted with:
  $ python plot_func.py
The total number of function evaluations are printed to stdout.
