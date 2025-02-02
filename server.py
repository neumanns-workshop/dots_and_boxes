from typing import Tuple
import random, threading, time, os
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, Add, Subtract, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

from flask import Flask, render_template
from flask_socketio import SocketIO
import logging

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add to global scope
training_state = {
    'paused': False,
    'speed': 1.0,
    'thread': None  # Track our training thread
}

class DotsAndBoxesEnv:
    """
    A generalized Dots & Boxes environment for an n x n grid of boxes.
    That means (n+1) x (n+1) dots, horizontal_count = (n+1)*n, vertical_count = n*(n+1).
    We'll track edges with an array: -1 => not drawn, 0 => player 0, 1 => player 1, etc.
    """
    def __init__(self, n=4):
        self.n = n
        self.horizontal_count = (n + 1) * n
        self.vertical_count = n * (n + 1)
        self.total_edges = self.horizontal_count + self.vertical_count
        self.num_boxes = n * n
        self.reset()
        # Precompute each box's 4 edges for fast box ownership checks
        self.box_map = self._compute_box_map()

    def _compute_box_map(self):
        """
        Creates a dict: box_index -> tuple(edge_indices).
        For box i,j => index = i*n + j
          - top = row i, col j..(j+1) [horizontal]
          - bottom = row i+1, col j..(j+1)
          - left = col j, row i..(i+1) [vertical], etc.
        But we rely on the known indexing of horizontal and vertical edges:
           * Horizontal edges: row*r + c => row in [0..n], c in [0..(n-1)]
           * Vertical edges: offset by horizontal_count, row in [0..(n-1)], c in [0..n]
        """
        box_map = {}
        # horizontal edge index = row*(n) + col for row in [0..n], col in [0..(n-1)]
        # vertical edge index   = horizontal_count + row*(n+1) + col
        for i in range(self.n):       # box rows
            for j in range(self.n):   # box cols
                box_idx = i * self.n + j
                top = i * self.n + j
                bottom = (i + 1) * self.n + j
                left = self.horizontal_count + i * (self.n + 1) + j
                right = left + 1
                box_map[box_idx] = (top, bottom, left, right)
        return box_map

    def reset(self):
        self.edges = [-1] * self.total_edges
        self.boxes = [None] * self.num_boxes
        self.scores = [0, 0]
        return self.get_state()

    def get_state(self):
        """
        The DQN only needs to know which edges are drawn or not.
        We'll return 1.0 if drawn, 0.0 if not drawn.
        """
        return np.array([1.0 if e != -1 else 0.0 for e in self.edges], dtype=np.float32)

    def get_legal_moves(self):
        return [i for i, owner in enumerate(self.edges) if owner == -1]

    def step(self, action: int, player: int) -> Tuple[np.ndarray, float, bool, bool]:
        """Execute one step in the environment.
        
        Args:
            action: Edge index to claim
            player: Player ID (0 or 1)
            
        Returns:
            Tuple of (next_state, reward, extra_turn, done)
        """
        if self.edges[action] != -1:
            raise Exception("Illegal move! Edge already taken.")
        self.edges[action] = player
        boxes_completed = self.check_boxes(player)
        reward = boxes_completed
        extra_turn = (boxes_completed > 0)
        done = all(e != -1 for e in self.edges)
        return self.get_state(), reward, extra_turn, done

    def check_boxes(self, player):
        completed = 0
        for i, owner in enumerate(self.boxes):
            if owner is not None:
                continue
            top, bottom, left, right = self.box_map[i]
            if (self.edges[top] != -1 and
                self.edges[bottom] != -1 and
                self.edges[left] != -1 and
                self.edges[right] != -1):
                self.boxes[i] = player
                self.scores[player] += 1
                completed += 1
        return completed

# Basic DQN agent
class DQNAgent:
    def __init__(self, state_size, action_size, player_id, **params):
        self.state_size = state_size
        self.action_size = action_size
        self.player_id = player_id
        self.gamma = params.get('gamma', 0.95)
        self.learning_rate = params.get('learning_rate', 0.001)
        self.batch_size = params.get('batch_size', 32)
        self.target_update_freq = params.get('target_update_freq', 100)
        self.memory = PrioritizedReplayBuffer(maxlen=2000)
        self.train_step = 0

        self.model = self._build_dueling_network()
        self.target_model = self._build_dueling_network()
        self.update_target_model()

        self.model_path = f"models/player_{player_id}_model.keras"

    def _build_dueling_network(self):
        """Build dueling network architecture."""
        state_input = Input(shape=(self.state_size,))
        
        # Shared layers
        hidden1 = Dense(64, activation='relu')(state_input)
        hidden2 = Dense(64, activation='relu')(hidden1)
        
        # Value stream
        value_hidden = Dense(32, activation='relu')(hidden2)
        value = Dense(1)(value_hidden)
        
        # Advantage stream
        advantage_hidden = Dense(32, activation='relu')(hidden2)
        advantage = Dense(self.action_size)(advantage_hidden)
        
        # Combine value and advantage
        output = Add()([
            value,
            Subtract()([
                advantage,
                Lambda(lambda x: K.mean(x, axis=1, keepdims=True))(advantage)
            ])
        ])
        
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, legal_moves):
        # Remove epsilon-greedy logic since we're using noisy networks
        q_values = self.model.predict(np.array([state]), verbose=0)[0]
        # Mask out illegal moves
        masked_q = [q_values[a] if a in legal_moves else -np.inf for a in range(self.action_size)]
        return int(np.argmax(masked_q))

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
            
        # Sample batch with priorities
        samples, indices, weights = self.memory.sample(self.batch_size)
        
        states = np.array([s[0] for s in samples])
        actions = np.array([s[1] for s in samples])
        rewards = np.array([s[2] for s in samples])
        next_states = np.array([s[3] for s in samples])
        dones = np.array([s[4] for s in samples])
        
        # Compute target Q values
        target_q_values = self.target_model.predict(next_states, verbose=0)
        max_target_q = np.max(target_q_values, axis=1)
        targets = rewards + (1 - dones) * self.gamma * max_target_q
        
        # Get current Q values and update only the chosen actions
        current_q = self.model.predict(states, verbose=0)
        td_errors = np.abs(targets - current_q[np.arange(self.batch_size), actions])
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Train model with importance sampling weights
        for i in range(self.batch_size):
            current_q[i][actions[i]] = targets[i]
        
        self.model.fit(
            states, 
            current_q, 
            sample_weight=weights,
            epochs=1, 
            verbose=0
        )
        
        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.update_target_model()

    def save_model(self, path):
        """Save the model to disk."""
        self.model.save(path)
        
    def load_model(self, path):
        """Load the model from disk if it exists."""
        try:
            self.model = tf.keras.models.load_model(path)
            self.target_model = tf.keras.models.load_model(path)
            print(f"Loaded existing model for player {self.player_id}")
            return True
        except:
            print(f"No existing model found for player {self.player_id}")
            return False

class PrioritizedReplayBuffer:
    def __init__(self, maxlen=2000, alpha=0.6, beta=0.4):
        """Initialize Prioritized Experience Replay buffer.
        
        Args:
            maxlen: Maximum size of buffer
            alpha: How much prioritization to use (0 = uniform sampling)
            beta: Importance sampling correction factor
        """
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta = beta
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        
    def append(self, experience):
        """Add new experience with max priority."""
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.buffer.append(experience)
        self.priorities.append(max_priority)
        
    def sample(self, batch_size):
        """Sample batch of experiences based on priorities."""
        if len(self.buffer) < batch_size:
            return []
            
        # Convert priorities to probabilities
        probs = np.array(self.priorities) ** self.alpha
        probs /= probs.sum()
        
        # Sample indices based on priorities
        indices = np.random.choice(
            len(self.buffer), 
            batch_size, 
            p=probs
        )
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** -self.beta
        weights /= weights.max()
        
        samples = [self.buffer[idx] for idx in indices]
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        """Update priorities for given indices."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-5  # Small constant for stability
            
    def __len__(self):
        return len(self.buffer)

@socketio.on('set_training_state')
def handle_training_state(data):
    training_state['paused'] = data['paused']
    logger.info(f"Training {'paused' if data['paused'] else 'resumed'}")

@socketio.on('set_training_speed')
def handle_training_speed(data):
    training_state['speed'] = data['speed']
    logger.info(f"Training speed set to {data['speed']}x")

def save_checkpoint(agent, episode):
    """Save model with episode number and ensure directory exists."""
    os.makedirs("models", exist_ok=True)
    path = f"models/player_{agent.player_id}_ep{episode}_model.keras"
    agent.save_model(path)
    logger.info(f"Saved checkpoint for player {agent.player_id} at episode {episode}")

def training_loop():
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    env = DotsAndBoxesEnv(n=4)
    state_size = env.total_edges
    action_size = env.total_edges
    
    # Initialize agents and try to load existing models
    agent0 = DQNAgent(state_size, action_size, player_id=0)
    agent1 = DQNAgent(state_size, action_size, player_id=1)
    agents = [agent0, agent1]
    
    # Try to load existing models
    for agent in agents:
        agent.load_model(f"models/player_{agent.player_id}_model.keras")
    
    episodes = 1000  # Increased for better learning
    checkpoint_frequency = 100  # Save every 100 episodes
    eval_frequency = 50  # Evaluate every 50 episodes
    
    # Track metrics for visualization
    metrics = {
        'episode_rewards': [],
        'win_rates': [[], []],  # [blue_rates[], green_rates[]]
        'eval_scores': [[], []],  # [blue_scores[], green_scores[]]
        'avg_q_values': [[], []],  # [blue_q_values[], green_q_values[]]
        'move_quality': [],  # Overall move quality
        'episodes': []
    }
    
    logger.info("Starting new training session")
    
    # Define checkpoint episodes for different difficulties
    checkpoint_episodes = {
        'easy': 1000,    # Basic strategy
        'medium': 5000,  # Intermediate play
        'hard': 10000    # Advanced tactics
    }
    
    for ep in range(episodes):
        episode_start = time.time()
        
        while training_state['paused']:
            time.sleep(0.1)  # Use time.sleep instead of socketio.sleep
            continue  # Skip the rest of the loop while paused
            
        state = env.reset()
        episode_q_values = [[], []]  # Track Q-values for this episode
        episode_moves = []  # Track move quality
        current_player = 0
        done = False
        
        while not done:
            legal_moves = env.get_legal_moves()
            agent = agents[current_player]
            action = agent.act(state, legal_moves)

            try:
                next_state, reward, extra_turn, done = env.step(action, current_player)
            except Exception as e:
                print("Error:", e)
                break

            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            # Always emit game state, but control the delay
            socketio.emit('game_state', {
                'edges': env.edges,
                'boxes': [b if b is not None else -1 for b in env.boxes],
                'scores': env.scores,
                'episode': ep,
                'current_player': current_player,
                'last_action': action,
                'n': env.n,
                'is_terminal': done  # Add terminal state
            })
            
            # Adjust sleep time based on speed
            if training_state['speed'] <= 0.2:
                time.sleep(2.0)  # Very slow - time to think
            elif training_state['speed'] <= 1.0:
                time.sleep(1.0 / training_state['speed'])  # Normal range
            elif training_state['speed'] <= 3.0:
                # Skip some frames for faster speeds
                if random.random() < (1.0 / training_state['speed']):
                    time.sleep(0.2)  # Brief pause for visualization
            else:
                # Training mode - minimal visualization
                if random.random() < 0.2:  # Show only 20% of moves
                    time.sleep(0.1)

            if not extra_turn:
                current_player = 1 - current_player
            state = next_state
            
            # Track Q-values and move quality
            q_values = agent.model.predict(np.array([state]), verbose=0)[0]
            episode_q_values[current_player].append(np.mean(q_values))
            best_q = np.max([q_values[m] for m in legal_moves])
            chosen_q = q_values[action]
            episode_moves.append(best_q - chosen_q)
        
        # Save models at specific episodes for difficulty levels
        if ep + 1 in checkpoint_episodes.values():
            for agent in agents:
                save_checkpoint(agent, ep + 1)
        
        # Save periodic checkpoints
        if ep % checkpoint_frequency == 0:
            for agent in agents:
                save_checkpoint(agent, ep + 1)
        
        # Single evaluation per checkpoint
        if ep % eval_frequency == 0:
            eval_scores = evaluate_agents(env, agents)
            metrics['eval_scores'][0].append(eval_scores[0])
            metrics['eval_scores'][1].append(eval_scores[1])
            metrics['episodes'].append(ep)
            
            logger.info(
                f"Episode {ep} - Eval: Blue={eval_scores[0]:.2f}, "
                f"Green={eval_scores[1]:.2f}"
            )
        
        # Update metrics
        metrics['avg_q_values'][0].append(float(np.mean(episode_q_values[0]) if episode_q_values[0] else 0))
        metrics['avg_q_values'][1].append(float(np.mean(episode_q_values[1]) if episode_q_values[1] else 0))
        metrics['move_quality'].append(float(np.mean(episode_moves) if episode_moves else 0))
        
        # Calculate win rates
        total_boxes = int(sum(env.scores))
        if total_boxes > 0:
            metrics['win_rates'][0].append(float(env.scores[0] / total_boxes))
            metrics['win_rates'][1].append(float(env.scores[1] / total_boxes))
        else:
            metrics['win_rates'][0].append(0.5)
            metrics['win_rates'][1].append(0.5)
        
        logger.info(
            f"Episode {ep:4d} | "
            f"Duration: {time.time() - episode_start:.2f}s | "
            f"Blue: {env.scores[0]:2d} | "
            f"Green: {env.scores[1]:2d} | "
            f"Win Rate: {metrics['win_rates'][0][-1]:.2f} | "
            f"Speed: {training_state['speed']:.1f}x"
        )
        
        socketio.emit('training_metrics', {
            'episode': int(ep),  # ensure it's a native int
            'scores': [int(s) for s in env.scores],  # convert to native ints
            'win_rates': [float(metrics['win_rates'][0][-1]), float(metrics['win_rates'][1][-1])],
            'eval_scores': [float(metrics['eval_scores'][0][-1]), float(metrics['eval_scores'][1][-1])] if ep % eval_frequency == 0 else None,
            'eval_episode': int(ep) if ep % eval_frequency == 0 else None,
            'avg_q_values': [
                float(metrics['avg_q_values'][0][-1]),
                float(metrics['avg_q_values'][1][-1])
            ],
            'move_quality': float(metrics['move_quality'][-1])
        })
        
        # Emit metrics less frequently when sped up
        base_sleep = 0.1
        adjusted_sleep = base_sleep / training_state['speed']
        time.sleep(adjusted_sleep)  # Use time.sleep for consistency

    # Save final models
    for agent in agents:
        save_checkpoint(agent, episodes)
        
    print("Training complete!")

def evaluate_agents(env, agents, num_games=10):
    """Evaluate agents without exploration or learning."""
    scores = [0, 0]
    for _ in range(num_games):
        state = env.reset()
        current_player = 0
        done = False
        
        while not done:
            agent = agents[current_player]
            legal_moves = env.get_legal_moves()
            # Use the model deterministically during evaluation
            with tf.device('/cpu:0'):  # Force CPU to avoid GPU warnings
                q_values = agent.model.predict(np.array([state]), verbose=0)[0]
            action = legal_moves[np.argmax([q_values[a] for a in legal_moves])]
            
            next_state, reward, extra_turn, done = env.step(action, current_player)
            if not extra_turn:
                current_player = 1 - current_player
            state = next_state
            
        scores[0] += env.scores[0]
        scores[1] += env.scores[1]
    
    return [s/num_games for s in scores]  # Return average scores

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('list_checkpoints')
def handle_list_checkpoints():
    checkpoints = []
    if not os.path.exists('models'):
        socketio.emit('checkpoints_list', checkpoints)
        return
        
    for file in os.listdir('models'):
        if file.startswith('player_') and file.endswith('_model.keras'):
            try:
                # Extract episode number
                ep_str = file.split('ep')[1].split('_')[0]
                episode = int(ep_str)
                
                checkpoints.append({
                    'player': file.split('_')[1],
                    'episode': episode,
                    'path': file
                })
            except:
                continue
    
    # Sort by episode number
    checkpoints.sort(key=lambda x: x['episode'])
    socketio.emit('checkpoints_list', checkpoints)

class HumanGame:
    def __init__(self, checkpoint_path):
        self.env = DotsAndBoxesEnv(n=4)
        self.state = self.env.reset()
        self.ai_agent = DQNAgent(self.env.total_edges, self.env.total_edges, player_id=1)
        self.ai_agent.load_model(checkpoint_path)
        self.current_player = 0  # Human starts
        self.game_over = False

# Store active human games
human_games = {}

@socketio.on('start_human_game')
def handle_start_human_game(data):
    game_id = str(random.randint(0, 1000000))
    game = HumanGame(f"models/{data['checkpoint']}")
    human_games[game_id] = game
    
    socketio.emit('human_game_started', {
        'game_id': game_id,
        'state': game.state.tolist(),
        'legal_moves': game.env.get_legal_moves()
    })

@socketio.on('human_move')
def handle_human_move(data):
    game = human_games.get(data['game_id'])
    if not game or game.game_over:
        return
    
    # Process human move
    move = data['move']
    if move not in game.env.get_legal_moves():
        return
    
    state, reward, extra_turn, done = game.env.step(move, 0)  # Human is player 0
    
    # If game not over and no extra turn, let AI move
    while not done and not extra_turn:
        game.current_player = 1
        ai_move = game.ai_agent.act(state, game.env.get_legal_moves())
        state, reward, extra_turn, done = game.env.step(ai_move, 1)
        if not extra_turn or done:
            game.current_player = 0
    
    game.game_over = done
    game.state = state
    
    # Send updated game state
    socketio.emit('game_update', {
        'game_id': data['game_id'],
        'state': state.tolist(),
        'edges': game.env.edges,
        'boxes': [b if b is not None else -1 for b in game.env.boxes],
        'scores': game.env.scores,
        'current_player': game.current_player,
        'legal_moves': game.env.get_legal_moves(),
        'game_over': game.game_over
    })

if __name__ == '__main__':
    if training_state['thread'] is None or not training_state['thread'].is_alive():
        training_state['thread'] = threading.Thread(target=training_loop, daemon=True)
        training_state['thread'].start()
    
    socketio.run(app, debug=False)  # Disable debug mode
