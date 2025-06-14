# main.py
#
# IRC Channel Operator Simulator
# Amended to use the Hugging Face transformers library for local inference.
#
# Required libraries:
# pip install curses-windows (if on Windows)
# For GPU support (NVIDIA), install PyTorch with CUDA:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Then install transformers:
# pip install transformers
#
# Note: The first time you run this, it will download the AI model, which may take a few moments.

import curses
import time
import random
import textwrap
import os
import sys
import torch

# Defer import of transformers to provide a loading message first
pipeline = None

class LLM_Generator:
    """
    Handles all interactions with the local Hugging Face model.
    """
    def __init__(self):
        """
        Initializes the text generation pipeline from Hugging Face.
        This will download the model on the first run.
        """
        global pipeline
        if pipeline is None:
            # This is imported here so we can show a loading screen in main()
            # before this heavy import happens.
            from transformers import pipeline as hf_pipeline
            
            # Check for a CUDA-enabled GPU and set the device accordingly.
            # device=0 for the first GPU, device=-1 for CPU.
            device = 0 if torch.cuda.is_available() else -1
            
            try:
                # Using a small, conversational model suitable for local execution
                self.generator = hf_pipeline(
                    'text-generation',
                    model='microsoft/DialoGPT-small',
                    device=device # Specify the device (GPU or CPU)
                )
            except Exception as e:
                # This fallback will be caught by the UI
                raise RuntimeError(
                    "Failed to load Hugging Face model. "
                    "Please check your internet connection and ensure 'transformers' and a "
                    "CUDA-compatible version of 'torch' are installed. "
                    f"Error: {e}"
                )

    def generate_message(self, user, chat_history):
        """
        Generates a message for a given user based on their persona and chat history.
        """
        # Create a simplified history string for the model's context
        # The model performs best with a simple conversational format.
        last_messages = "\n".join([f"{msg['username']}: {msg['text']}" for msg in chat_history[-4:]])
        
        # We give the model the persona, the last few messages, and prompt it to respond as the user.
        prompt = (
            f"This is a chat conversation. {user.username}'s personality is: {user.persona}\n"
            f"{last_messages}\n"
            f"{user.username}:"
        )

        try:
            # Generate a response, keeping it short.
            outputs = self.generator(
                prompt,
                max_new_tokens=35,  # Limit the length of the generated response
                num_return_sequences=1,
                pad_token_id=self.generator.tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
            )
            
            # The model output includes the prompt, so we need to clean it up.
            full_text = outputs[0]['generated_text']
            # Remove the original prompt from the generated text
            response = full_text.replace(prompt, '').strip()
            # Take only the first line of the response
            return response.split('\n')[0]

        except Exception:
            # Fallback in case of a model error
            return "..."

class User:
    """
    Represents an AI user in the chat channel.
    """
    def __init__(self, username, persona):
        self.username = username
        self.persona = persona
        self.behavior_score = 70  # Starts at a neutral 70 out of 100
        self.is_op = False
        self.banned = False
        self.kick_timer = 0

    def misbehave(self):
        """
        Reduces behavior score, making the user more likely to misbehave again.
        """
        self.behavior_score = max(0, self.behavior_score - 15)

    def behave(self):
        """
        Increases behavior score.
        """
        self.behavior_score = min(100, self.behavior_score + 10)


class UI:
    """
    Handles all the curses-based screen drawing and input.
    """
    def __init__(self, stdscr, game):
        self.stdscr = stdscr
        self.game = game
        curses.start_color()
        # Define color pairs
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK) # Default
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK) # Operator
        curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK) # System
        curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK) # Warning/Error
        curses.init_pair(5, curses.COLOR_GREEN, curses.COLOR_BLACK) # Status bar
        self.stdscr.nodelay(True)  # Non-blocking input

    def draw(self, messages, status, input_buffer):
        """
        Draws the entire UI to the screen.
        """
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()

        # 1. Draw Status Bar
        status_bar_text = (f"Channel: #ChatZone | Op: {self.game.player_op_name} | "
                           f"Health: {self.game.channel_health:.0f}% | Score: {self.game.score}")
        self.stdscr.attron(curses.color_pair(5))
        self.stdscr.addstr(0, 0, status_bar_text.ljust(width))
        self.stdscr.attroff(curses.color_pair(5))


        # 2. Draw Main Chat Window
        chat_height = height - 3
        chat_win = self.stdscr.subwin(chat_height, width, 1, 0)
        chat_win.box()
        
        # Display messages, wrapping long lines
        y_pos = chat_height - 2
        for msg in reversed(messages):
            if y_pos < 1:
                break
            
            color = curses.color_pair(1) # Default user color
            if msg['type'] == 'operator':
                color = curses.color_pair(2)
            elif msg['type'] == 'system':
                color = curses.color_pair(3)
                
            full_message = f"<{msg['username']}> {msg['text']}"
            lines = textwrap.wrap(full_message, width - 4) # Wrap text
            
            for line in reversed(lines):
                 if y_pos < 1:
                    break
                 chat_win.addstr(y_pos, 2, line, color)
                 y_pos -=1


        # 3. Draw Input Line
        self.stdscr.addstr(height - 1, 0, f"> {input_buffer}")

        self.stdscr.refresh()

    def get_input(self):
        """
        Gets a single character of input without blocking.
        """
        try:
            return self.stdscr.getkey()
        except curses.error:
            return None


class Game:
    """
    The main class that runs the game loop and manages state.
    """
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.player_op_name = "@player"
        self.ui = UI(self.stdscr, self)

        try:
            self.llm_generator = LLM_Generator()
        except RuntimeError as e:
            # Handle model loading failure gracefully
            self.show_error_and_exit(str(e))
            # The program should exit after the error message
            sys.exit(1)


        self.messages = []
        self.input_buffer = ""
        self.channel_health = 100
        self.score = 0
        self.users = self._create_user_personas()
        self.running = True
        
        self.add_system_message("Welcome to IRC Channel Operator Simulator!")
        self.add_system_message("This version uses a local AI model (no API key needed).")
        device_name = "NVIDIA GPU" if torch.cuda.is_available() else "CPU"
        self.add_system_message(f"AI model is running on: {device_name}")
        self.add_system_message("Type /help for a list of commands.")


    def _create_user_personas(self):
        personas = {
            "CoolDude23": "is chill and uses a lot of slang. They are mostly harmless.",
            "InfoSeeker": "asks a lot of questions, sometimes off-topic.",
            "ArgumentBot": "loves to argue and contradict others.",
            "Spammer101": "posts repetitive or nonsensical messages.",
            "TheGamer": "only wants to talk about video games.",
            "DramaQueen": "tries to start drama and get attention.",
            "HelpfulHannah": "tries to be helpful and answer questions.",
            "SilentBob": "rarely talks, but when he does, it's cryptic."
        }
        return [User(name, persona) for name, persona in personas.items()]

    def run(self):
        """
        The main game loop.
        """
        last_update = time.time()
        ai_message_timer = time.time()

        while self.running and self.channel_health > 0:
            current_time = time.time()

            # Handle player input
            key = self.ui.get_input()
            if key:
                self.process_input(key)

            # Update timers for kicked users
            for user in self.users:
                if user.kick_timer > 0:
                    user.kick_timer -= (current_time - last_update)

            # AI user speaks periodically
            if current_time - ai_message_timer > random.uniform(3, 6):
                self.ai_speak()
                ai_message_timer = current_time

            # Redraw UI
            self.ui.draw(self.messages, self.get_status(), self.input_buffer)
            
            last_update = current_time
            time.sleep(0.1) # Small delay to prevent high CPU usage

        # Game over
        if self.channel_health <= 0:
            self.add_system_message("Channel health reached 0%. The channel has descended into chaos. GAME OVER.", 'error')
        self.ui.draw(self.messages, self.get_status(), "GAME OVER. Press any key to exit.")
        self.stdscr.nodelay(False)
        self.stdscr.getch()

    def process_input(self, key):
        """
        Processes a single key press from the player.
        """
        if key in ("KEY_BACKSPACE", '\b', '\x7f'):
            self.input_buffer = self.input_buffer[:-1]
        elif key == '\n':  # Enter key
            self.execute_command(self.input_buffer)
            self.input_buffer = ""
        elif key and len(key) == 1 and len(self.input_buffer) < self.stdscr.getmaxyx()[1] - 4:
            self.input_buffer += key
            
    def execute_command(self, command_str):
        """
        Parses and executes a command from the player.
        """
        if not command_str:
            return
        
        if command_str.startswith('/'):
            parts = command_str.split(' ')
            cmd = parts[0].lower()
            args = parts[1:]

            if cmd == '/say':
                self.add_message(self.player_op_name, ' '.join(args), 'operator')
            elif cmd == '/kick':
                self.handle_kick(args)
            elif cmd == '/ban':
                self.handle_ban(args)
            elif cmd == '/warn':
                self.handle_warn(args)
            elif cmd == '/op':
                self.handle_op(args)
            elif cmd == '/help':
                self.show_help()
            elif cmd == '/quit':
                self.running = False
            else:
                self.add_system_message(f"Unknown command: {cmd}", 'error')
        else:
             # Default action is to say the message
            self.add_message(self.player_op_name, command_str, 'operator')
            
    def handle_kick(self, args):
        if not args:
            self.add_system_message("Usage: /kick <username>", 'error')
            return
        username = args[0]
        user = self.find_user(username)
        if user:
            if user.banned:
                self.add_system_message(f"User {username} is already banned.", 'error')
                return

            self.add_system_message(f"*** {self.player_op_name} has kicked {username}.")
            user.kick_timer = 20 # 20 second timeout before they can rejoin
            
            # Scoring
            if user.behavior_score < 40: # They were likely misbehaving
                self.score += 10
                self.add_system_message(f"Good kick! +10 points.", 'system')
            else:
                self.score -= 5
                self.add_system_message(f"Unjustified kick! -5 points.", 'system')
        else:
            self.add_system_message(f"User not found: {username}", 'error')

    def handle_ban(self, args):
        if not args:
            self.add_system_message("Usage: /ban <username>", 'error')
            return
        username = args[0]
        user = self.find_user(username)
        if user:
            user.banned = True
            self.add_system_message(f"*** {self.player_op_name} has banned {username}.")
            if user.behavior_score < 25:
                 self.score += 25
                 self.add_system_message(f"Justice served! +25 points.", 'system')
            else:
                self.score -= 10
                self.add_system_message(f"That was harsh... -10 points.", 'system')
        else:
            self.add_system_message(f"User not found: {username}", 'error')

    def handle_warn(self, args):
        if not args:
            self.add_system_message("Usage: /warn <username>", 'error')
            return
        username = args[0]
        user = self.find_user(username)
        if user:
            self.add_system_message(f"*** {username} has been warned by the operator.")
            user.behave() # Warning improves their behavior score
        else:
            self.add_system_message(f"User not found: {username}", 'error')
            
    def handle_op(self, args):
        if not args:
            self.add_system_message("Usage: /op <username>", 'error')
            return
        username = args[0]
        user = self.find_user(username)
        if user:
            if not user.is_op:
                user.is_op = True
                self.add_system_message(f"*** {self.player_op_name} has given operator status to {username}.")
            else:
                self.add_system_message(f"{username} is already an operator.", 'error')
        else:
            self.add_system_message(f"User not found: {username}", 'error')

    def show_help(self):
        self.add_system_message("--- Available Commands ---")
        self.add_system_message("/say <msg> - Send a message.")
        self.add_system_message("/warn <user> - Warn a user.")
        self.add_system_message("/kick <user> - Kick a user from the channel.")
        self.add_system_message("/ban <user> - Ban a user permanently.")
        self.add_system_message("/op <user> - Make a user a sub-operator.")
        self.add_system_message("/quit - Exit the game.")

    def ai_speak(self):
        """
        Selects an AI user to generate and post a message.
        """
        eligible_users = [u for u in self.users if not u.banned and u.kick_timer <= 0]
        if not eligible_users:
            return

        user_to_speak = random.choice(eligible_users)
        
        # Chance to misbehave based on behavior score
        is_spam = False
        if random.randint(0, 100) > user_to_speak.behavior_score:
            is_spam = True
            user_to_speak.misbehave()
            self.channel_health = max(0, self.channel_health - 2)

        message_text = self.llm_generator.generate_message(user_to_speak, self.messages)
        
        # Sub-ops might automatically warn spammers
        if is_spam:
            sub_ops = [u for u in self.users if u.is_op and u != user_to_speak]
            if sub_ops and random.random() < 0.3: # 30% chance for a sub-op to act
                op_actor = random.choice(sub_ops)
                self.add_message(op_actor.username, f"Whoa, chill out {user_to_speak.username}!", 'operator')
                self.channel_health = min(100, self.channel_health + 1) # Mitigate damage
                return # The sub-op's action pre-empts the spam

        self.add_message(user_to_speak.username, message_text)
        
        # Slowly recover health if chat is peaceful
        if not is_spam and self.channel_health < 100:
            self.channel_health = min(100, self.channel_health + 0.5)


    def add_message(self, username, text, msg_type='user'):
        """Adds a message to the chat history."""
        self.messages.append({'username': username, 'text': text, 'type': msg_type})
        if len(self.messages) > 100:  # Keep history from getting too long
            self.messages.pop(0)

    def add_system_message(self, text, level='system'):
        color_map = {'system': 'system', 'error': 'error'}
        self.add_message('*SYSTEM*', text, msg_type=color_map.get(level, 'system'))

    def find_user(self, username):
        """Finds a user by their username (case-insensitive)."""
        for user in self.users:
            if user.username.lower() == username.lower():
                return user
        return None

    def get_status(self):
        """Returns the current game status."""
        return {
            "channel_health": self.channel_health,
            "score": self.score,
        }
        
    def show_error_and_exit(self, message):
        """Displays an error message in the center of the screen and waits for a keypress to exit."""
        self.stdscr.nodelay(False)
        self.stdscr.clear()
        height, width = self.stdscr.getmaxyx()
        
        # Use color pair 4 (red) for the error message
        error_color = curses.color_pair(4) | curses.A_BOLD
        
        lines = textwrap.wrap(message, width - 6)
        start_y = height // 2 - len(lines) // 2
        for i, line in enumerate(lines):
            self.stdscr.addstr(start_y + i, (width - len(line)) // 2, line, error_color)
            
        exit_prompt = "Press any key to exit..."
        self.stdscr.addstr(height - 2, (width - len(exit_prompt)) // 2, exit_prompt)
        self.stdscr.refresh()
        self.stdscr.getch()
        self.running = False


def main(stdscr):
    """The main entry point of the application."""
    # Check if terminal supports colors
    if not curses.has_colors():
        print("Your terminal does not support colors.")
        sys.exit(1)

    # Check for terminal size
    height, width = stdscr.getmaxyx()
    if height < 20 or width < 80:
        # Exit curses mode to print the message
        curses.endwin()
        print("Please resize your terminal to at least 80x20 and try again.")
        sys.exit(1)
        
    # Show loading message before heavy imports
    stdscr.clear()
    loading_msg = "Loading AI model, please wait... (this may take a moment)"
    stdscr.addstr(height // 2, (width - len(loading_msg)) // 2, loading_msg)
    stdscr.refresh()
    
    try:
        # Now we can initialize the game, which triggers the model download
        game = Game(stdscr)
        game.run()
    except SystemExit:
        # This allows show_error_and_exit to work correctly
        pass
    except Exception as e:
        # Gracefully exit curses mode and print the error
        curses.endwin()
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except Exception as e:
        # This catches errors during curses initialization
        print(f"Failed to start the game: {e}")
