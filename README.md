`ggsolver` is a Python package to model, manipulate, and solve 
games and hypergames on graphs. 

[Note: This package is still in development and not all features are implemented yet. 
The available algorithms are marked with check-marks below.]


## Unique Features

There exist several tools to model and solve games on graphs (e.g., PRISM, stormpy). 
Why do we need `ggsolver`?

1. **Hypergame support:** 
    A hypergame is game of (potentially evolving) perceptual games of each player. 
    `ggsolver` optimizes the modeling of hypergames on graphs by efficiently managing the perceptual games.
2. **Integration with ML-frameworks:** 
    `ggsolver` is designed to be integrated with ML-frameworks (e.g., PyTorch, TensorFlow) to enable the use of 
    learning techniques for solving (hyper)games on graphs.
3. **Incompatible solvers:** 
    It is either tedious or not straightforward to model non-trivial transition models into PRISM, 
    where next state is determined by complex if-else clauses.     


## Algorithms 
The package implements / interfaces several known algorithms for solving 
games on graphs, including:
* Sure/almost-sure winning in deterministic two-player turn-based games 
  (Zielonka's strategy construction algorithm)
* Sure/almost-sure winning in stochastic two-player turn-based games 
* Sure winning in stochastic two-player concurrent games 
* Almost-sure winning in stochastic two-player concurrent games

New/novel algorithms for two-player adversarial deception games:
* Deceptive sure winning in dynamic hypergames on graphs under action misperception [Link]()
* Stealthy deceptive sure winning in static hypergames on graphs under labeling misperception [Link]()
* Stealthy deceptive almost-sure winning in static hypergames on graphs under labeling misperception [Link]()
* Mechanism (game graph) design for optimal stealthy deception in static hypergames on graphs under labeling misperception [Link]()
* Behavioral subjectively rationalizability in dynamic hypergames on graphs under specification misperception [Link]()

New/novel algorithms for preference-based planning in MDP:
* Safe and positively improving strategies in MDP with incomplete preferences over temporal objectives [Link]()
* Safe and almost-surely improving strategies in MDP with incomplete preferences over temporal objectives [Link]()
* ✅ Non-dominated strategies in MDP with incomplete preferences over temporal objectives [Link]()

New/novel algorithms for two/multi-player games with partial preferences:
* Nash equilibrium in two-player games with adversarial preferences [Link]()
* Privacy-aware Nash equilibrium in two-player games with  [Link]()
* Admissibility in multi-player coalition games with incomplete preferences [Link]()




# Installation Instructions

## PRISM Games (Windows)

Download and follow installation instructions from the following link:
https://www.prismmodelchecker.org/games/download.php 

Ensure JRE v9+ is available to run PRISM and PRISM-games.

> Note: If the `load_prism` function may returns  `java.lang.ClassNotFoundException: prism.PrismCL` error, then 
> modify the `\PATH\TO\PRISM-GAMES-INSTALLATION\bin\prism.bat` as suggested by https://stackoverflow.com/questions/70800624/error-could-not-find-or-load-main-class-prism-prismcl-caused-by-java-lang-clas
> The modified `prism.bat` file is provided in the `external` directory, which can be copied to prism-games' bin folder. 


# How to setup and solve a game? 

1. Decide how to model the game: 
    transition system: by defining `S, A, T, AP, L, ...` components
    module: by defining transition system similar to PRISM modules, but implemented in Python
    PRISM model: using a `.prism` file.
2. From `templates/` folder, copy the template file to implement the abstract `TSys, Module, PRISMGame` classes.
3. Implement the relevant abstract methods of chosen model class.
4. Select a flattened representation -- `GraphGame, MatrixGame, BDDGame` and create an instance.
5. Call the `build` method on the game instance to flatten the game into its representation. 
    Otherwise, leave it as it is if you will compute a symbolic product game etc.
6. Define objectives, using any logic or automaton representation user wants.
7. In case of automata-theoretic approach, the product must be implemented by user, unless 
    it is already available for the class of games considered.
8. The solvers are implemented in folders of various class of game.