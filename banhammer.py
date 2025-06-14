import curses
import time
import random
import google.generativeai as genai
import textwrap
import os
import sys

class LLM_Generator:
    """
    Handles all interactions with the Google Generative AI API.
    """
    def __init__(self, api_key):
        self.api_key = api_key
        try:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        except Exception as e:
            # This is a fallback for the UI to catch if the API key is invalid
            raise ValueError(f"Failed to configure Gemini API: {e}")

    def generate_message(self, user, chat_history):
        """
        Generates a message for a given user based on their persona and chat history.
        """
        # Create a simplified history for the prompt
        last_messages = "\n".join([f"<{msg['username']}> {msg['text']}" for msg in chat_history[-5:]])

        prompt = (
            f"You are an AI user in an IRC chat room simulation. Your name is {user.username} "
            f"and your personality is: \"{user.persona}\".\n"
            f"The last few messages in the channel were:\n{last_messages}\n"
            f"What is your short, single-line response in the style of a 90s/early 2000s chat user? "
            f"Keep it under 15 words. Behave according to your personality. Your message should just be the text, without your username."
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip().replace("\n", " ")
        except Exception:
            # Fallback in case of API error
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
                           f"Health: {self.game.channel_health}% | Score: {self.game.score}")
        self.stdscr.attron(curses.color_pair(5))
        self.stdscr.addstr(0, 0, status_bar_text.ljust(width))
        self.stdscr.attroff(curses.color_pair(5))


        # 2. Draw Main Chat Window
        chat_height = height - 3
        chat_win = self.stdscr.subwin(chat_height, width, 1, 0)
        chat_win.border()
        
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
            
    def get_api_key_from_user(self):
        """
        Prompts the user for their API key before the main game starts.
        """
        self.stdscr.nodelay(False) # Blocking input for this part
        api_key = ""
        height, width = self.stdscr.getmaxyx()
        while True:
            self.stdscr.clear()
            self.stdscr.addstr(height // 2 - 1, (width - 40) // 2, "Enter your Google AI Studio API Key:")
            self.stdscr.addstr(height // 2, (width - 40) // 2, "> " + api_key)
            self.stdscr.refresh()
            key = self.stdscr.getch()
            if key == 10: # Enter
                if api_key:
                    break
            elif key == curses.KEY_BACKSPACE or key == 127:
                api_key = api_key[:-1]
            elif 32 <= key <= 126:
                api_key += chr(key)
        self.stdscr.nodelay(True) # Back to non-blocking
        return api_key.strip()


class Game:
    """
    The main class that runs the game loop and manages state.
    """
    def __init__(self, stdscr):
        self.stdscr = stdscr
        self.player_op_name = "@player"
        # Prompt for API key before initializing other components
        self.ui = UI(self.stdscr, self) # Pass self to UI
        self.api_key = self.ui.get_api_key_from_user()
        try:
            self.llm_generator = LLM_Generator(self.api_key)
        except ValueError as e:
            # Handle invalid API key gracefully
            self.show_error_and_exit(str(e))

        self.messages = []
        self.input_buffer = ""
        self.channel_health = 100
        self.score = 0
        self.users = self._create_user_personas()
        self.running = True
        
        self.add_system_message("Welcome to IRC Channel Operator Simulator!")
        self.add_system_message("Type /help for a list of commands.")


    def _create_user_personas(self):
        personas = {
            "CoolDude23": "Chill and uses a lot of slang. Mostly harmless.",
            "InfoSeeker": "Asks a lot of questions, sometimes off-topic.",
            "ArgumentBot": "Loves to argue and contradict others.",
            "Spammer101": "Posts repetitive or nonsensical messages.",
            "TheGamer": "Only wants to talk about video games.",
            "DramaQueen": "Tries to start drama and get attention.",
            "HelpfulHannah": "Tries to be helpful and answer questions.",
            "SilentBob": "Rarely talks, but when he does, it's cryptic."
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
            if current_time - ai_message_timer > random.uniform(2, 5):
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
        elif key and len(key) == 1:
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
        lines = textwrap.wrap(message, width - 4)
        start_y = height // 2 - len(lines) // 2
        for i, line in enumerate(lines):
            self.stdscr.addstr(start_y + i, (width - len(line)) // 2, line, curses.color_pair(4))
        self.stdscr.addstr(height - 2, (width - 25) // 2, "Press any key to exit...")
        self.stdscr.refresh()
        self.stdscr.getch()
        self.running = False


def main(stdscr):
    """The main entry point of the application."""
    # Check if terminal supports colors
    if not curses.has_colors():
        sys.exit("Your terminal does not support colors.")
        return

    # Check for terminal size
    height, width = stdscr.getmaxyx()
    if height < 20 or width < 80:
        sys.exit("Please resize your terminal to at least 80x20 and try again.")
        return
        
    try:
        game = Game(stdscr)
        game.run()
    except Exception as e:
        # Gracefully exit curses mode and print the error
        curses.endwin()
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # Initialize curses
    try:
        curses.wrapper(main)
    except SystemExit as e:
        print(e)
    except Exception as e:
        print(f"Failed to start the game: {e}")

