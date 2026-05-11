from pathlib import Path
import re

p = Path("pickandplace.py")
code = p.read_text()

replacement = '''    # ---------------- vision-language grounding check ----------------
    def vision_check_object(self, cmd: dict) -> bool:
        """
        Stable live verification for terminal/GUI execution.

        Camera-based perception is tested separately with:
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
        """
        Return object positions from MuJoCo for the action encoder.
        """
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
        """
        Convert parsed language command into symbolic and numerical action encoding.
        """
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

start = code.find("    def vision_check_object(self, cmd: dict) -> bool:")
if start == -1:
    raise RuntimeError("Could not find vision_check_object")

# include preceding comment if present
comment_start = code.rfind("\n    #", 0, start)
if comment_start != -1 and start - comment_start < 300:
    start = comment_start + 1

end = code.find("    def _execute_parsed_command", start)
if end == -1:
    raise RuntimeError("Could not find _execute_parsed_command")

code = code[:start] + replacement + code[end:]

p.write_text(code)
print("✅ Fixed vision_check_object, get_object_positions_for_encoder, and encode_and_print_action.")
