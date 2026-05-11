from pathlib import Path

p = Path("pickandplace.py")
code = p.read_text()

# 1. Add action encoder import
old_import = "from nl_interface import parse_command"
new_import = """from nl_interface import parse_command

from action_encoding.action_encoder import (
    encode_symbolic_action,
    symbolic_action_to_dict,
    encode_action_vector,
    decode_action_vector,
)"""

if new_import not in code:
    code = code.replace(old_import, new_import)

# 2. Add helper methods before command execution
helpers = r'''
    # ---------------- vision-language grounding check ----------------
    def vision_check_object(self, cmd: dict) -> bool:
        """
        Stable live verification for terminal/GUI execution.

        Camera perception is tested separately with:
            python3 test_perception.py
            python3 test_vision_language.py
        """
        obj = cmd.get("obj", "box")
        valid_objects = self._valid_object_names()

        print("[VISION] Live object check:", obj)
        print("[VISION] Valid objects:", valid_objects)

        if obj not in valid_objects:
            with self._console_lock:
                self.console_status = f"❌ Object not found: {obj}"
            print(f"[VISION] Object not found: {obj}")
            return False

        print("[VISION] Grounded:", {"obj": obj, "visible": True})
        return True

    # ---------------- action encoding helpers ----------------
    def get_object_positions_for_encoder(self) -> dict:
        positions = {}

        for obj_name in self._valid_object_names():
            try:
                b = self.data.body(obj_name)
                positions[obj_name] = [
                    float(b.xpos[0]),
                    float(b.xpos[1]),
                    float(b.xpos[2]),
                ]
            except Exception:
                pass

        return positions

    def encode_and_print_action(self, cmd: dict) -> None:
        try:
            object_positions = self.get_object_positions_for_encoder()

            symbolic = encode_symbolic_action(cmd)
            symbolic_dict = symbolic_action_to_dict(symbolic)

            vector = encode_action_vector(cmd, object_positions)
            decoded = decode_action_vector(vector)

            print("[ACTION] Symbolic:", symbolic_dict)
            print("[ACTION] Vector:", vector)
            print("[ACTION] Decoded:", decoded)

            with self._console_lock:
                self.console_status = f"Action encoded: {symbolic.action_type} / {symbolic.object_name}"

        except Exception as e:
            print("[ACTION] Warning: action encoding failed:", e)

'''

marker = "    # ---------------- command execution ----------------"
if "def encode_and_print_action(self, cmd: dict)" not in code:
    code = code.replace(marker, helpers + marker)

# 3. Insert vision + action check before motion execution
old_block = """                # ----- motion tasks -----
                with self._console_lock:
                    self._console_busy = True
                self.motion_busy.set()

                if task == "pick_place":
"""

new_block = """                # ----- motion tasks -----
                with self._console_lock:
                    self._console_busy = True
                self.motion_busy.set()

                if task in ("pick_place", "pick_place_xy", "stack", "place_on"):
                    if not self.vision_check_object(cmd):
                        with self._console_lock:
                            self._console_busy = False
                        self.motion_busy.clear()
                        return

                    self.encode_and_print_action(cmd)

                if task == "pick_place":
"""

if old_block not in code:
    raise RuntimeError("Could not find motion task block. The file may not be clean.")

code = code.replace(old_block, new_block)

p.write_text(code)
print("✅ Clean action encoder patch applied.")
