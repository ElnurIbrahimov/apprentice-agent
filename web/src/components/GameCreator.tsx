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
    ]
  },
];

const GAME_SYSTEM_PROMPT = `You are an expert HTML5 game developer. Generate a complete, playable game in a single HTML file.

Rules:
- Output ONLY the complete HTML code starting with <!DOCTYPE html>
- Use HTML5 Canvas for rendering (create a properly sized canvas element)
- Include ALL game logic in a <script> tag
- The game MUST be fully playable immediately on load
- Implement proper game loop with requestAnimationFrame
- Add keyboard AND touch/click controls for mobile compatibility
- Include: start screen, gameplay, game over screen with score
- Add sound effects using Web Audio API (simple tones, not files)
- Use smooth animations and particle effects where appropriate
- Include score tracking and high score in localStorage
- Make the canvas responsive (fill screen on mobile, centered on desktop)
- Add proper collision detection
- NO external dependencies, NO images (use shapes/CSS art), NO CDN
- NO markdown fences, NO explanation text, ONLY the HTML document`;

export function GameCreator() {
  return (
    <WebCreator
      creatorMode="game"
      customTemplates={GAME_TEMPLATES}
      customSystemPrompt={GAME_SYSTEM_PROMPT}
    />
  );
}
