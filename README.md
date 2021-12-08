# Efficient learning 
The repository shows an Efficient learning method using critical points of a response surface.

To run the script in TableI and TableII you need to install mystic and its requirements:  https://github.com/uqfoundation/mystic

# mystic
highly-constrained non-convex optimization and uncertainty quantification

#


Adjustable settings are in _model.py, but are set at values used in manuscript.
Results may be slightly different due to randomness from sampler and optimizer.

Executes with:
  $ python main_workflow.py
and may take anywhere from a few minutes to a few days, depending on the model
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
```ruby 
  import dill
  cost = dill.load(open('cost.pkl', 'rb'))
  ```
while relevant results from the databases are plotted with:
```ruby 
python plot_func.py
```
and test score convergence is plotted with:
```ruby 
python plot_*_converge.py    (* = loose, tight)
  ```
with "loose" corresponding to loose tolerance, and tight to strict tolerance.  

=======

