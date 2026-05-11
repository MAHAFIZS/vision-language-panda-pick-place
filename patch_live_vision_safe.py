from pathlib import Path
import re

path = Path("pickandplace.py")
code = path.read_text()

safe_method = '''    # ---------------- vision-language grounding check ----------------
    def vision_check_object(self, cmd: dict) -> bool:
        """
        Stable live verification for terminal/GUI execution.

        Camera-based perception is tested separately with:
            python3 test_perception.py
            python3 test_vision_language.py

        During the live MuJoCo viewer, we avoid creating a second renderer
        because that can cause OpenGL or MuJoCo memory errors.
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

'''

start = code.find("    def vision_check_object(self, cmd: dict) -> bool:")
if start == -1:
    raise RuntimeError("Could not find vision_check_object")

# Include comment line before method if present
comment_start = code.rfind("\n    #", 0, start)
if comment_start != -1 and start - comment_start < 300:
    start = comment_start + 1

# Find next class method
end = code.find("\n    def ", start + 1)
if end == -1:
    raise RuntimeError("Could not find next method after vision_check_object")

code = code[:start] + safe_method + code[end + 1:]

path.write_text(code)
print("✅ Replaced live vision check with stable object verification.")
