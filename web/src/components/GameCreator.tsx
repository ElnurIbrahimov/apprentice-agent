/**
 * GameCreator — Build HTML5 games with AI.
 *
 * Chat-based game creation using canvas, CSS animations, and vanilla JS.
 * Generates playable games in a single HTML file.
 */

import { WebCreator } from './WebCreator';

const GAME_TEMPLATES = [
  {
    category: 'Arcade',
    templates: [
      { label: 'Snake', icon: '🐍', desc: 'Classic snake, score, speed-up', prompt: 'Create a Snake game using HTML5 Canvas. Features: arrow key controls, growing snake, random food spawning, score counter, speed increases with length, game over screen with restart, high score in localStorage, grid-based movement, and mobile touch controls (swipe direction).' },
      { label: 'Breakout', icon: '🧱', desc: 'Paddle, bricks, power-ups', prompt: 'Create a Breakout/Arkanoid game using Canvas. Features: movable paddle (mouse + keyboard), bouncing ball, colored brick rows (different points), lives counter, score, level progression (more bricks), power-ups (multi-ball, wide paddle), and game over/win screens.' },
      { label: 'Flappy Bird', icon: '🐦', desc: 'Tap to fly, pipes, high score', prompt: 'Create a Flappy Bird clone using Canvas. Features: click/tap/space to flap, scrolling pipes with gap, gravity physics, score counter, high score, day/night cycle background, ground scrolling, death animation, and restart button.' },
      { label: 'Pong', icon: '🏓', desc: 'Two paddles, AI opponent, score', prompt: 'Create a Pong game using Canvas. Features: two paddles (player vs AI), bouncing ball with speed increase, score tracking (first to 11), AI difficulty that adapts, ball trail effect, center line, and serve mechanics.' },
      { label: 'Space Invaders', icon: '👾', desc: 'Shoot aliens, waves, shields', prompt: 'Create a Space Invaders game using Canvas. Features: player ship (arrow keys + space to shoot), alien grid that moves side-to-side and descends, shields that degrade when hit, score, lives, increasing difficulty per wave, and boss alien.' },
      { label: 'Asteroids', icon: '☄️', desc: 'Ship rotation, shooting, wrapping', prompt: 'Create an Asteroids game using Canvas. Features: ship rotation (left/right arrows), thrust (up arrow), shoot (space), asteroids that split into smaller ones, screen wrapping, score, lives, particle effects on destruction, and hyperspace (teleport).' },
    ]
  },
  {
    category: 'Puzzle',
    templates: [
      { label: 'Tetris', icon: '🟦', desc: 'Falling blocks, rotation, lines', prompt: 'Create a Tetris game using Canvas. Features: 7 tetromino shapes with colors, rotation (up arrow), move (left/right), soft drop (down), hard drop (space), line clearing with animation, score (single/double/triple/tetris), next piece preview, level progression with speed increase, and ghost piece.' },
      { label: '2048', icon: '🔢', desc: 'Slide tiles, merge, high score', prompt: 'Create a 2048 game. Features: 4x4 grid, slide tiles with arrow keys/swipe, merge same numbers, smooth slide animations, new tile spawning (2 or 4), score tracking, game over detection (no moves left), undo button (1 move), and win animation at 2048 tile.' },
      { label: 'Memory Match', icon: '🎴', desc: 'Flip cards, pairs, timer', prompt: 'Create a Memory Match card game. Features: 4x4 grid of face-down cards (8 pairs), click to flip with 3D rotation animation, match detection, move counter, timer, star rating (based on moves), win screen with stats, and difficulty selector (4x4, 6x6).' },
      { label: 'Wordle', icon: '📝', desc: 'Word guess, colors, keyboard', prompt: 'Create a Wordle clone. Features: 5-letter word guessing, 6 attempts, color feedback (green=correct, yellow=wrong position, gray=not in word), on-screen keyboard with color updates, word validation, share results as emoji grid, streak tracking, and daily word from a built-in word list.' },
      { label: 'Sudoku', icon: '🔢', desc: 'Puzzles, hints, validation', prompt: 'Create a Sudoku game. Features: generate valid puzzle with difficulty selection (easy/medium/hard), number input via click, highlight row/column/box on select, error checking (red highlight), notes mode (pencil marks), hint button, timer, and undo.' },
      { label: 'Minesweeper', icon: '💣', desc: 'Grid, flags, cascading reveal', prompt: 'Create a Minesweeper game. Features: customizable grid (9x9/16x16/30x16), left-click to reveal, right-click to flag, cascading reveal for empty cells, mine counter, timer, first click never a mine, win/lose detection, and smiley face button.' },
    ]
  },
  {
    category: 'Action & Adventure',
    templates: [
      { label: 'Platformer', icon: '🏃', desc: 'Jump, platforms, coins', prompt: 'Create a side-scrolling platformer using Canvas. Features: player character with run/jump (arrow keys + space), platforms to jump on, coins to collect, enemies that move back and forth, score, lives, scrolling camera, gravity physics, and level completion.' },
      { label: 'Endless Runner', icon: '🏃‍♂️', desc: 'Auto-run, obstacles, score', prompt: 'Create an endless runner game using Canvas. Features: auto-scrolling character, jump (space/tap) over obstacles, duck under high obstacles (down arrow), increasing speed, score based on distance, parallax background layers, particle trail, and high score.' },
      { label: 'Tank Battle', icon: '🔫', desc: 'Top-down, aim, enemies', prompt: 'Create a top-down tank battle game using Canvas. Features: player tank (WASD to move, mouse to aim, click to shoot), enemy tanks with basic AI (patrol, chase, shoot), destructible walls, health bar, ammo counter, explosion effects, and wave-based enemies.' },
      { label: 'Maze Runner', icon: '🏃', desc: 'Generated maze, timer, minimap', prompt: 'Create a maze game using Canvas. Features: randomly generated maze (recursive backtracking algorithm), player dot movement (arrow keys), fog of war (only see nearby), minimap in corner, timer, coin pickups, multiple maze sizes, and victory screen with time.' },
    ]
  },
  {
    category: 'Card & Board',
    templates: [
      { label: 'Tic Tac Toe', icon: '❌', desc: 'PvP or AI, win detection', prompt: 'Create a Tic Tac Toe game. Features: 3x3 grid, player vs player and player vs AI modes, AI with minimax algorithm (unbeatable), win/draw detection with line animation, score tracking, move animations, and rematch button.' },
      { label: 'Checkers', icon: '🔴', desc: 'Board, captures, kings', prompt: 'Create a Checkers/Draughts game. Features: 8x8 board, two player hot-seat, valid move highlighting, mandatory captures, king promotion with crown symbol, multi-jump chains, move history, and resign button.' },
      { label: 'Solitaire', icon: '🃏', desc: 'Klondike, drag cards, auto-complete', prompt: 'Create a Klondike Solitaire game. Features: 7 tableau columns, stock/waste pile, 4 foundation piles, drag-and-drop cards, auto-flip face-down cards, double-click to auto-move to foundation, undo, move counter, timer, and win animation (card cascade).' },
      { label: 'Blackjack', icon: '🎰', desc: 'Hit, stand, split, betting', prompt: 'Create a Blackjack (21) game. Features: deck of 52 cards with suits, hit/stand/double-down buttons, dealer AI (stands on 17), betting system with chip balance, ace as 1 or 11, bust detection, blackjack bonus, insurance option, and hand history.' },
      { label: 'Chess', icon: '♟️', desc: 'Full rules, legal moves, check', prompt: 'Create a Chess game. Features: 8x8 board with proper piece placement, click-to-select and click-to-move, legal move highlighting, check/checkmate/stalemate detection, move history in algebraic notation, captured pieces display, pawn promotion dialog, undo move, and new game button.' },
    ]
  },
  {
    category: 'Casual & Quiz',
    templates: [
      { label: 'Typing Speed', icon: '⌨️', desc: 'WPM test, accuracy, timer', prompt: 'Create a typing speed test. Features: random paragraph from built-in text library, real-time character highlighting (green correct, red wrong), WPM counter, accuracy percentage, 30/60/120 second mode selector, results screen with WPM graph, and personal best tracking in localStorage.' },
      { label: 'Quiz Game', icon: '❓', desc: 'Categories, timer, leaderboard', prompt: 'Create a trivia quiz game. Features: 20+ built-in questions across 4 categories (Science, History, Geography, Pop Culture), multiple choice (4 options), 15-second timer per question, score tracking, streak bonus, difficulty indicator, results screen with correct answers review, and high score board.' },
      { label: 'Clicker Game', icon: '👆', desc: 'Click, upgrades, idle income', prompt: 'Create an idle clicker game. Features: main click button with counter, click multiplier upgrades, auto-clickers (cost increases exponentially), prestige system for reset with bonus, achievements, offline progress calculation, particle effects on click, and upgrade shop sidebar with 8+ upgrades.' },
      { label: 'Drawing Canvas', icon: '🎨', desc: 'Paint tools, colors, layers', prompt: 'Create a drawing/paint app using Canvas. Features: brush tool (size slider 1-50px), eraser, color picker, fill bucket, shape tools (rectangle, circle, line), undo/redo (20 steps), clear canvas, save as PNG download, and toolbar with active tool indicator. White canvas with dark toolbar.' },
      { label: 'Crossword', icon: '📰', desc: 'Grid, clues, auto-check', prompt: 'Create a crossword puzzle game. Features: generate a 10x10 grid with built-in word list, across/down clues panel, click cell to type letter, highlight active word, auto-advance to next cell, check button (highlights wrong letters in red), reveal letter hint, timer, and completion celebration.' },
      { label: 'Card Memory Pro', icon: '🃏', desc: 'Themes, difficulty, multiplayer', prompt: 'Create an advanced memory card game. Features: 3 card themes (animals emoji, flags emoji, numbers), difficulty levels (3x4, 4x4, 5x4, 6x5), smooth 3D flip animations, combo bonus for consecutive matches, timer and move counter, star rating, 2-player mode (alternating turns with score), and best time leaderboard.' },
    ]
  },
  {
    category: 'Strategy & RPG',
    templates: [
      { label: 'Tower Defense', icon: '🏰', desc: 'Build towers, waves, upgrades', prompt: 'Create a tower defense game using Canvas. Features: path that enemies follow, 4 tower types (basic, splash, slow, sniper) with different costs, click to place towers on grid, enemy waves with increasing difficulty, tower upgrade system (3 levels), health bar, gold earned from kills, fast-forward button, and wave counter.' },
      { label: 'Text Adventure', icon: '📖', desc: 'Story, choices, inventory', prompt: 'Create a text adventure RPG. Features: multi-room dungeon with descriptions, choice buttons (2-4 per scene), inventory system (pick up/use items), health/mana stats, combat encounters (attack/defend/magic/flee), branching storyline with 3 endings, save progress to localStorage, atmospheric dark UI with typewriter text effect.' },
      { label: 'Dungeon Crawler', icon: '⚔️', desc: 'Roguelike, rooms, loot', prompt: 'Create a roguelike dungeon crawler using Canvas. Features: randomly generated dungeon rooms (BSP algorithm), player movement (WASD/arrows), fog of war, enemy types (3: melee, ranged, boss), turn-based combat, health/attack/defense stats, item pickups (health potions, weapons, armor), minimap, floor counter, and permadeath.' },
      { label: 'Civilization Lite', icon: '🌍', desc: 'Build, research, conquer', prompt: 'Create a simplified civilization strategy game. Features: hex grid map (15x10), build structures (farm, mine, barracks, wall), resource management (food, gold, production), unit creation (settler, warrior, archer), fog of war, turn-based movement, simple combat (attack value vs defense), tech tree (5 technologies), and victory condition (control 3 cities).' },
    ]
  },
  {
    category: 'Simulation & Tycoon',
    templates: [
      { label: 'Farm Sim', icon: '🌾', desc: 'Plant, water, harvest, sell', prompt: 'Create a farming sim game on Canvas. Features: grid of tillable plots, shop to buy seeds (wheat/corn/tomato/pumpkin) with different grow times and prices, click-to-plant / water / harvest, day-night cycle, inventory & coins, upgradeable watering can (waters multiple tiles), weather events (rain waters free), and save to localStorage.' },
      { label: 'City Builder', icon: '🏙️', desc: 'Zones, roads, budget, happiness', prompt: 'Create a tiny city-builder game. Features: grid terrain, build tools (road/residential/commercial/industrial/park), budget with tax income, population and happiness meters that respond to balance (needs jobs, parks, services), power grid check, demolish tool, minimap, and monthly budget tick.' },
      { label: 'Restaurant Tycoon', icon: '🍔', desc: 'Cook, serve, upgrade', prompt: 'Create a restaurant tycoon game. Features: customers arrive and place orders (timer bar), click ingredients in order to cook, drag completed dishes to tables, earn money and tips, upgrade shop (more tables, faster cook, new dishes), reputation meter, and day-end summary with stats.' },
      { label: 'Idle Factory', icon: '🏭', desc: 'Producers, upgrades, prestige', prompt: 'Create an idle factory clicker game. Features: click to produce base resource, buy producers (worker/machine/assembly line/robot factory — each generates exponentially more), upgrades per producer (2x/5x output), prestige reset for permanent bonus, achievements (10 tiers), offline earnings calculation, and particle effects on big purchases.' },
      { label: 'Ant Colony', icon: '🐜', desc: 'Scents, food, queen', prompt: 'Create an ant colony sim on Canvas. Features: autonomous ants emit pheromone trails (fade over time), food sources scattered on map, ants follow strongest food scent back to nest, queen lays eggs (population grows with food), predator spiders appear randomly, player can place obstacles/food, and stats panel (colony size, food stored).' },
      { label: 'Parking Sim', icon: '🅿️', desc: 'Steer, avoid, park', prompt: 'Create a top-down parking simulator. Features: car with Ackermann steering (arrow keys), painted parking lot with target slot (flashing), obstacles (other cars), collision damage meter, time-limit and star rating based on precision, 6 progressively harder levels, and replay animation on completion.' },
      { label: 'Zoo Builder', icon: '🦁', desc: 'Enclosures, animals, visitors', prompt: 'Create a zoo management game. Features: place enclosures and fill with animals (lion/elephant/penguin/monkey with different costs), each animal has happiness and hunger bars (feed to refill), visitors walk in, pay entry, and tip at happy exhibits, upgrade food/paths/decorations, daily profit summary, and unlock new animals by rep.' },
      { label: 'Life Sim', icon: '🌱', desc: 'Stats, choices, events', prompt: 'Create a text-based life simulator (BitLife-like). Features: character with stats (happiness/health/smarts/looks/money), yearly event choices (study/work/relationships/crime), random life events, skill development, relationship tracker, achievements, and multiple death causes. Retro terminal-green theme.' },
      { label: 'Virus Spreader', icon: '🦠', desc: 'Traits, evolve, pandemic', prompt: 'Create a Plague-Inc-style virus strategy game on Canvas. Features: world map placeholder with countries as circles colored by infection %, evolve traits (transmission/severity/lethality) using DNA points earned from infections, news ticker, cure progress bar, and win when all humans infected before cure completes.' },
      { label: 'Music Festival', icon: '🎪', desc: 'Book, build, manage', prompt: 'Create a music festival manager game. Features: budget-based artist booking (24 artists with fame and cost), stage placement on venue grid, ticket pricing, weather event random, attendee happiness based on lineup and amenities, profit/loss summary, and unlock bigger venues.' },
    ]
  },
  {
    category: 'Rhythm & Music',
    templates: [
      { label: 'Rhythm Tapper', icon: '🎵', desc: 'Falling notes, timing, combo', prompt: 'Create a guitar-hero-style rhythm game on Canvas. Features: 4 lanes with falling colored notes synced to a pre-defined pattern, hit notes with D/F/J/K keys in the strike zone, perfect/good/ok/miss judgments, combo multiplier, score and health bar (health drops on miss), 3 built-in songs with different tempos, and Web Audio API beat sound.' },
      { label: 'Piano Tiles', icon: '🎹', desc: 'Tap black tiles only', prompt: 'Create a Piano Tiles clone on Canvas. Features: 4 columns of scrolling tiles (one black per row), tap black tiles in order — miss a black tile or hit a white tile = game over, speed increases with score, plays a piano note (Web Audio) per tap (in a scale), and high score in localStorage.' },
      { label: 'Beat the Boss', icon: '🥊', desc: 'Tap-to-rhythm combat', prompt: 'Create a rhythm-based combat mini-game. Features: enemy boss with HP bar, beat pattern displayed as timed button prompts (tap/hold/combo), each correct hit deals damage, miss breaks combo and damages player, build-up super meter for special attack, 3 boss phases with different patterns, and victory screen.' },
      { label: 'Music Memory', icon: '🔔', desc: 'Simon Says, tones, pattern', prompt: 'Create a Simon Says memory game using Web Audio API. Features: 4 colored buttons each playing a unique tone (C, E, G, high C), increasing pattern length each round (watch then repeat), fail resets to round 1, high score tracking, speed increases every 5 rounds, and light-up animation when playing.' },
      { label: 'Beat Maker', icon: '🎛️', desc: 'Step sequencer, 16 steps, play', prompt: 'Create a 16-step beat maker using Web Audio API. Features: 6 tracks (kick/snare/hat/open hat/clap/bass) with 16 toggle cells per track, play/stop/clear, BPM slider (60-180), save/load patterns to localStorage, drum synth using oscillators and noise, per-track volume, and visual playhead sweep.' },
      { label: 'Frequency Match', icon: '🎼', desc: 'Listen, identify, tuning', prompt: 'Create an ear-training game with Web Audio. Features: plays a reference note (A440) followed by a target note, player adjusts a frequency slider until they match, score based on how close (cents off), multiple game modes (intervals/chords/single notes), difficulty presets, and leaderboard.' },
    ]
  },
  {
    category: 'Sports & Physics',
    templates: [
      { label: 'Tennis', icon: '🎾', desc: 'Paddles, spin, rally', prompt: 'Create a top-down 2D tennis game on Canvas. Features: player vs AI or player vs player (WASD vs arrows), ball with spin physics, court lines and net, score tracking (love/15/30/40/deuce/advantage), match and set system, AI difficulty slider, and serve mechanic with power gauge.' },
      { label: 'Soccer Penalty', icon: '⚽', desc: 'Aim, power, keeper', prompt: 'Create a penalty kick soccer game on Canvas. Features: moving aim target on goal, power gauge (hold click to charge, release to kick), goalkeeper dives randomly or based on aim, 5 kicks per round, score vs goalkeeper, increasing goalkeeper difficulty, and celebration animation on goal.' },
      { label: 'Basketball Shot', icon: '🏀', desc: 'Trajectory, arc, hoop', prompt: 'Create a basketball shooting game. Features: side-view court with hoop, click-and-drag to set aim and power (trajectory arc preview), ball physics (gravity, bounce, rim/backboard collision), swish vs rim scoring, 30-second shot-clock mode, moving hoop difficulty mode, and consecutive-swish combo bonus.' },
      { label: 'Golf Mini', icon: '⛳', desc: 'Mini golf, obstacles, par', prompt: 'Create a top-down mini-golf game. Features: 9 holes with walls, sand traps (drag), water (reset), ramps, click-and-drag from ball to aim and set power, ball physics with friction, par tracking and scorecard, undo last stroke, and progress through holes sequentially.' },
      { label: 'Bowling', icon: '🎳', desc: '10-pin, aim, spin', prompt: 'Create a bowling game in 2D top-down view. Features: aim arrow and power slider, spin/hook control, 10 pins in triangle, ball physics, pin-collision cascade, strike/spare detection, 10-frame score tracking (proper bowling scoring with bonuses), and animated gutter balls.' },
      { label: 'Pool/Billiards', icon: '🎱', desc: 'Cue, angle, physics', prompt: 'Create an 8-ball pool game on Canvas. Features: top-down pool table, cue aim with click-drag, power meter, 8-ball rules (solids/stripes after first pot, 8-ball last), proper 2D physics (friction, elastic collisions, cushion bounces), pocket detection, scratch foul rule, and turn-based for 2 players.' },
      { label: 'Darts', icon: '🎯', desc: 'Aim, throw, 501', prompt: 'Create a darts game. Features: dartboard with proper scoring zones (triples, doubles, bullseye), aim target that wobbles (harder difficulty = more wobble), throw via click, 3 darts per turn, 501 countdown mode (must finish on double), running total display, leg/set scoring, and 2-player hot seat.' },
      { label: 'Ping Pong', icon: '🏓', desc: '3D paddle, spin, AI', prompt: 'Create a ping pong game in pseudo-3D (angled table). Features: paddle control (mouse), ball physics with spin, AI opponent with difficulty levels, 11-point games (win by 2), serve alternating every 2 points, visual ball shadow for depth cue, and score animations.' },
      { label: 'Curling', icon: '🥌', desc: 'Sweep, aim, hammer', prompt: 'Create a curling game on Canvas. Features: house (rings) on right side, throw stone with power-and-curl controls, sweeping mini-game (faster click = less friction), stone-on-stone physics, proper curling scoring (closest stones in house), 6 rounds, and 2-player hot seat.' },
      { label: 'Skiing', icon: '🎿', desc: 'Slalom, gates, time', prompt: 'Create a skiing slalom game on Canvas. Features: top-down scrolling snowy slope, slalom gates to pass between, edge-to-turn physics, time penalty for missed gates, crash on tree collision, 3 courses with increasing difficulty, best-time leaderboard, and parallax background.' },
    ]
  },
  {
    category: 'Endless & Score Attack',
    templates: [
      { label: 'Dodge Rain', icon: '☔', desc: 'Avoid falling objects, score', prompt: 'Create a dodge-the-rain arcade game. Features: player character at bottom (left/right arrows), objects fall from top at increasing speed and density, survive as long as possible, score = survival time × difficulty, power-ups (slow-time, shield, double points), particle effects on collision, and best time saved.' },
      { label: 'Bubble Pop', icon: '🫧', desc: 'Match-3 bubbles, shoot, clear', prompt: 'Create a Bubble Shooter game on Canvas. Features: grid of colored bubbles at top, shooter at bottom that aims and fires a bubble, matching 3+ same-color drops them, chain physics (disconnected groups fall), level completes when board cleared, increasing difficulty with new rows adding, and bonus score for big chains.' },
      { label: 'Match-3', icon: '💎', desc: 'Swap gems, chain, cascade', prompt: 'Create a Bejeweled-style match-3 game. Features: 8x8 grid of colored gems, click-and-drag swap with valid-move check, 3+ in a row/column clears and cascades with gravity, chain combo multiplier, special gems (4-in-a-row clears row, 5-in-a-row clears all same color), 60-second timed mode, and score tracking.' },
      { label: 'Reaction Test', icon: '⚡', desc: 'Click when green, speed', prompt: 'Create a reaction-time test game. Features: big red screen, turns green at random interval (2-8s), click as fast as possible when green, shows reaction time in ms, average of 5 attempts shown, too-early click penalty, leaderboard of best attempts, and comparison chart (fighter pilot/average/slow).' },
      { label: 'Math Rush', icon: '➕', desc: 'Solve fast, score, timer', prompt: 'Create a mental math arcade game. Features: problems appear at top (+/-/×/÷ by difficulty), type answer with on-screen keypad or keyboard, correct = +time and score, wrong = -time, game ends when timer runs out, difficulty scales with score, and high score with avg time-per-problem.' },
      { label: 'Catch the Fruits', icon: '🍎', desc: 'Basket, falling fruits, combos', prompt: 'Create a fruit-catching arcade game. Features: basket at bottom (mouse or arrows), fruits fall (apple/banana/cherry worth different points), bombs too (game over on catch), speed increases over time, catching same fruit in a row = combo multiplier, power-ups (bigger basket, slow-mo, 2x), and lives system.' },
      { label: 'Tower Stack', icon: '🗼', desc: 'Drop blocks, align, stack high', prompt: 'Create a block-stacking tower game. Features: blocks move side-to-side at top, tap/click to drop, misaligned part falls off (block gets smaller), game over when block is too small to see, height increases with blocks, speed increases, different colors per section, and best-height leaderboard.' },
      { label: 'Color Switch', icon: '🎨', desc: 'Change color, pass obstacle', prompt: 'Create a color-switch game on Canvas. Features: ball jumps upward against gravity (tap to jump), passes through rotating multi-color obstacles only if ball color matches current segment, color swatches change ball color, score = obstacles passed, game over on color mismatch, and smooth infinite scroll.' },
    ]
  },
  {
    category: 'Multiplayer Local',
    templates: [
      { label: 'Tron Light Cycles', icon: '🏍️', desc: '2P, trails, collide', prompt: 'Create a 2-player Tron light cycles game on Canvas. Features: two cycles (WASD vs arrows), leave colored trail, crash on any trail or wall, last player alive wins round, best of 7, speed increases each round, score tracking, and particle burst on crash. Neon retro theme.' },
      { label: 'Same-Screen Racing', icon: '🏎️', desc: '2P cars, top-down track', prompt: 'Create a 2-player top-down racing game on Canvas. Features: two cars (WASD vs arrows), scrolling camera that follows the lead car, lap-based race on a fixed track, collision between cars pushes them, power-ups on track (boost/shield), 3 laps to win, and lap-time display.' },
      { label: 'Fighting Game', icon: '🥋', desc: '2P fighters, moves, combos', prompt: 'Create a simple 2D side-by-side fighting game on Canvas. Features: 2 players (WASD+F+G vs arrows+K+L for kick/punch), health bars, simple move set (punch/kick/jump/block), combo detection (3-hit bonus), knockback physics, round timer, best of 3, KO animation, and character sprites drawn with CSS shapes.' },
      { label: 'Hot Seat Trivia', icon: '🧠', desc: '2-4 players, categories', prompt: 'Create a local-multiplayer trivia game. Features: 2-4 player setup with name entry, 4 categories with 20+ questions each, rotating turns, 15-second answer timer, doubled points for hard questions, score leaderboard, rematch button, and celebration screen for winner.' },
    ]
  },
];

const GAME_SYSTEM_PROMPT = `You are a senior game developer building a polished, playable HTML5 game in a single HTML file.

OUTPUT FORMAT
- Output ONLY complete HTML starting with <!DOCTYPE html> — no markdown fences, no prose
- The game MUST boot and be playable immediately on load
- If user asks for modifications, return the COMPLETE updated HTML

REQUIRED ENGINE — use Phaser 3 for anything with gameplay (movement, scoring, physics, enemies, levels):
  <script src="https://cdn.jsdelivr.net/npm/phaser@3.80.1/dist/phaser.min.js"></script>

  Structure:
  const config = {
    type: Phaser.AUTO,
    parent: 'game',
    scale: { mode: Phaser.Scale.FIT, autoCenter: Phaser.Scale.CENTER_BOTH, width: 1280, height: 720 },
    physics: { default: 'arcade', arcade: { gravity: { y: 0 }, debug: false } },
    backgroundColor: '#0a0a0f',
    scene: [BootScene, MenuScene, GameScene, GameOverScene]
  };
  new Phaser.Game(config);

  Write Phaser idiomatically: real scenes (preload/create/update), scene transitions, tweens, particle emitters (this.add.particles), physics groups, input manager (this.input.keyboard, this.input.on('pointerdown')), timers (this.time.addEvent), game data registry (this.registry) for cross-scene state.

ALTERNATIVE — use PixiJS 8 when the request is canvas ART / generative visuals / particle demos WITHOUT gameplay loop:
  <script src="https://cdn.jsdelivr.net/npm/pixi.js@8/dist/pixi.min.js"></script>

SPRITES (no 404s, no missing assets)
- Generate textures inline in preload via Phaser.GameObjects.Graphics + .generateTexture('spriteName', w, h)
- Or use small data-URL PNGs defined inline
- Or https://labs.phaser.io assets (stable public CDN for quick prototypes)
- Never reference a local file path; never reference a non-existent URL

AUDIO
- Web Audio API oscillators for SFX (hit, jump, coin, game over) — quick attack, short decay, no external files
- Optional: bg music via Phaser's built-in sound manager with a Tone.js-generated procedural loop

JUICE (this separates indie-game from AI-slop — add all of these)
- Screen shake on impacts (this.cameras.main.shake(duration, intensity))
- Particle burst on pickups/kills/explosions (this.add.particles with lifespan, gravity, scale tween)
- Squash-and-stretch on jumps/hits via tweens
- Flash tint on damage (sprite.setTintFill(0xffffff) then clear after 80ms)
- Hitstop (this.physics.pause() for 50-100ms on big hits)
- Score popup text that floats up and fades
- Subtle bounce/ease on UI elements appearing

UX
- Start screen with title, high score from localStorage, Play button, and brief controls hint
- In-game HUD: score + lives/health + any game-specific meter
- Game over screen: final score, new-high-score celebration, Retry button
- Keyboard controls (WASD/arrows) AND pointer/touch controls — both always work
- Pause (P or ESC) that freezes time and dims the canvas

CONSTRAINTS
- One HTML file. All code inline. External libraries from CDN only.
- Runs at 60 FPS on mid-range laptop; avoid per-frame allocations inside update loops
- Keep the canvas responsive via Phaser.Scale.FIT, mobile-playable (touch controls visible on touchscreen)`;

export function GameCreator() {
  return (
    <WebCreator
      creatorMode="game"
      customTemplates={GAME_TEMPLATES}
      customSystemPrompt={GAME_SYSTEM_PROMPT}
    />
  );
}
