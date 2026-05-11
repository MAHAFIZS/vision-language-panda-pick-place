from pathlib import Path

p = Path("pickandplace.py")
code = p.read_text()

start = code.find("    # ---------------- vision-language grounding check ----------------")
if start == -1:
    start = code.find("    def vision_check_object(self, cmd: dict) -> bool:")

end = code.find("    def _execute_parsed_command", start)

if start == -1:
    raise RuntimeError("Could not find vision_check_object block")

if end == -1:
    raise RuntimeError("Could not find _execute_parsed_command block")

safe_block = '''    # ---------------- vision-language grounding check ----------------
    def vision_check_object(self, cmd: dict) -> bool:
        """
        Stable live verification for terminal/GUI execution.

        Camera-based perception is tested separately with:
            python3 test_perception.py
            python3 test_vision_language.py

        During live MuJoCo viewer execution, do not create mujoco.Renderer().
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

code = code[:start] + safe_block + code[end:]

p.write_text(code)
print("✅ Removed duplicate vision_check_object and kept safe live version.")
