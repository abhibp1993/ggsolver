CHANGELOG 
========= 

## 0.1.2 

* Added new docker latest, devel images. Updated installation instructions in documentation.
* Added `is_isomorphic_to` function to Graph class. Updated docs and example.


## 0.1.3 

* Added `logic.pl` module with `simplify, evaluate, to_plformula` and `all_sat` methods for PL formulas.
* Moved `sigma` method to `Automaton` class with default implementation.
* Fixed the representation of `Automaton` acceptance conditions.
* Added new module: `automata`, with `DFA, Monitor, DBA, DCBA, DPA` classes. Docs and example added.
* Automaton can be constructed by passing components (Q, AP, Trans, q0, F) to the constructor.


## 0.1.4 

* Bugfix in `SpotAutomaton.is_semi_deterministic() function. 
* SpotAutomaton.acc_cond adheres to ggsolver.models.Automaton convention.
* Made init arguments to classes in ggsolver.automata optional.
* Added Automaton.from_automaton() function to construct DFA, DBA ... from SpotAutomaton.
* Added example for translating LTL to DFA.


## 0.1.5 

* Pointed graphify implemented for GraphicalModel. 
* SubGraph class added. 
* Solvers now operate on SubGraph on given graph to construct node_winner, edge_winner properties.
* GraphicalModel.states() is now REQUIRED to return a list of hashable objects.
* dtptb package added for deterministic two-player turn-based games.
* SWin, ASWin for algorithms for reachability and safety added.
* mdp package added for qualitative Markov decision processes. ASWin, PWin algorithms for reachability added.
* Added progress bars to graphify and solvers in dtptb package.
* [Bugfix] input_domain stores the name of function (= graph property) that stores the input domain. 
  Thus, the reconstructed graphical model has the same input domain functions as the original model.


## 0.1.6 

* (logic) [Added] `atoms` parameter to SpotAutomaton constructor. 
* (logic) [Added] `Formula` base class to logic package. 
* (logic) [Added] `PL` (propositional logic) module to logic package.
* (logic) [Added] `simplify, sat2formula, allsat, evaluate, substitute` functions to `PL` class. 
* (logic) [Added] `LTL` logic to logic package.  
* (logic) [Added] `ScLTL` logic to logic package.
* (logic) [Added] `PL, LTL, ScLTL` classes are hashable. 
* (logic) [Added] `translate` method to `LTL, ScLTL` classes. `ScLTL` translates to `DFA` object, 
  `LTL` translates to `SpotAutomaton` object. 
* (logic) [Enhance] Automaton edges are labeled with simplified PL formulas. 
* (logic) [Enhance] Specialized `Automaton`'s graphify method using `PL` formulas as labels. 
* (logic) [Bugfix] Fixed the circular imports:  `models.Automaton` depends on `pl.PL`. Class `pl.PL` depends on `i_spot.SpotAutomaton`. 
  And class `i_spot.SpotAutomaton` depends on `models.Automaton`.
* (ggsolver core) [Bugfix] The `base_only` flag for `GraphicalModel.graphify` results in only underlying graph being constructed.  
* (ggsolver arch) [Enhance] Changed the import statements throughout ggsolver: `from ggsolver.**.** import **` -> `import ggsolver.**`


## [Unreleased]

* Documentation for changes in v0.1.6.  
* Reorganized the examples folder v0.1.6. 
* (logic) [Added] Parser for `PrefLTL` logic.
* (logic) [??] PrefLTL and PrefScLTL are hashable. 
* (logic) [Added] `PrefModel` construction given PrefScLTL formula using `Formula2Model` transformer. 
* (logic) [Added] `null_assumption` option to `PrefLTL` class to enforce the condition that
  "satisfying some formula is better than satisfying none."
* (logic) [Added] `PrefScLTL` to `DFPA` translation. 
* (example) [Added] an example to demo `PrefScLTL` to `DFPA` translation.
* (logic) [Algorithm] Ranking scheme on `DFPA`'s preference graph. 
* (logic) [Refactored] `inc_pbp` package containing safe and almost-sure/positive improving strategies. 
* (logic) [Added] Product of DFPA and MDP.
* (gridworld) [Designed] New architecture for pygame based gridworld simulator. 
* (gridworld) [Added] color_util module with color names. 
* (gridworld) [Added] `StateMachine` class to track progress of a game state. `StateMachine` can be 
  stepped forward as well as backward. The action, state history is stored (with ability to limit length of history).
* (gridworld) [Added] `Window` class that creates a new pygame window. Window handles event system, 
  rendering and state-machine updates.  
* (gridworld) [Added] `GWSim` class that maintains the state-machine corresponding to game and updates the window.
* (gridworld) [Added] `Control` class that represents any renderable element on Window. Each control handles a 
  set of events. 
* (gridworld) [Added] `Grid` control with ability to create and manipulate gridworlds.  
* (gridworld) [Added] `Cell` control representing a cell in a Grid control.  

