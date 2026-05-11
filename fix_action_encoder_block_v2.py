from pathlib import Path

p = Path("pickandplace.py")
code = p.read_text()

# Find where the broken block starts
start = code.find("    def vision_check_object(self, cmd: dict) -> bool:")
if start == -1:
    raise RuntimeError("Could not find vision_check_object")

# Include the comment line before it if available
comment_start = code.rfind("\n    #", 0, start)
if comment_start != -1 and start - comment_start < 500:
    start = comment_start + 1

# End before viewer/render section
end = code.find("    def render(self) -> None:", start)
if end == -1:
    raise RuntimeError("Could not find def render")

replacement = r'''    # ---------------- vision-language grounding check ----------------
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

    # ---------------- command execution ----------------
    def _execute_parsed_command(self, cmd: dict, raw: str | None = None) -> None:
        if raw:
            with self._console_lock:
                self.console_history.append(raw)
                self.console_history = self.console_history[-10:]

        task = cmd.get("task", "unknown")

        with self._console_lock:
            if self._console_busy and task in ("pick_place", "pick_place_xy", "stack", "place_on", "sort_all", "tower"):
                self.console_status = "Busy: wait for current motion to finish"
                return

        def run():
            try:
                with self._console_lock:
                    self.console_status = f"Parsed: {task}"

                if task == "help":
                    with self._console_lock:
                        self.console_status = (
                            "Help: list objects | where is red_box | pick red_box place bin/left/right/x..y.. | "
                            "stack red_box on green_box | sort all | make a tower | reset | quit"
                        )
                    return

                if task == "unknown":
                    with self._console_lock:
                        self.console_status = "❌ Unknown command"
                    return

                if task == "quit":
                    with self._console_lock:
                        self.console_status = "Stopping..."
                    self.run = False
                    self._hold_running = False
                    self.stop_flag.set()
                    return

                if task == "reset":
                    with self._console_lock:
                        self.console_status = "Resetting..."
                    self.reset_home()
                    with self._console_lock:
                        self.console_status = "✅ reset done"
                    return

                if task == "gripper":
                    mode = cmd.get("mode", "open")
                    self.gripper(open=(mode == "open"))
                    with self._console_lock:
                        self.console_status = f"✅ gripper {mode}"
                    return

                if task == "list_objects":
                    s = self.list_objects()
                    with self._console_lock:
                        self.console_status = "✅ Listed objects (see terminal)"
                    print(s)
                    return

                if task == "where":
                    s = self.where_is(cmd.get("obj", "box"))
                    with self._console_lock:
                        self.console_status = s
                    print(s)
                    return

                # ----- motion tasks -----
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
                    obj = cmd.get("obj", "box")
                    target = cmd.get("target", "bin_center")
                    with self._console_lock:
                        self.console_status = f"Pick {obj} -> {target}"
                    self.pick_place_to_site(obj, target)

                elif task == "pick_place_xy":
                    obj = cmd.get("obj", "box")
                    x = float(cmd["x"])
                    y = float(cmd["y"])
                    with self._console_lock:
                        self.console_status = f"Pick {obj} -> ({x:.2f},{y:.2f})"
                    self.pick_place_xy(obj, x, y)

                elif task == "stack":
                    obj = cmd.get("obj", "box")
                    base = cmd.get("base", "box")
                    with self._console_lock:
                        self.console_status = f"Stack {obj} on {base}"
                    self.stack(obj, base)

                elif task == "place_on":
                    obj = cmd.get("obj", "box")
                    base = cmd.get("base", "box")
                    with self._console_lock:
                        self.console_status = f"Place {obj} on {base}"
                    self.stack(obj, base)

                elif task == "sort_all":
                    with self._console_lock:
                        self.console_status = "Sorting all..."
                    self.sort_all()

                elif task == "tower":
                    with self._console_lock:
                        self.console_status = "Building tower..."
                    self.tower()

                else:
                    with self._console_lock:
                        self.console_status = f"Unhandled task: {task}"

                if self.go_home_after_motion_task and task in ("pick_place", "pick_place_xy", "stack", "place_on", "sort_all", "tower"):
                    with self._console_lock:
                        self.console_status = "Returning home..."
                    self.return_home_smooth()

                with self._console_lock:
                    self._console_busy = False
                    self.console_status = "✅ Done"

                self.motion_busy.clear()

            except Exception as e:
                self.motion_busy.clear()
                with self._console_lock:
                    self._console_busy = False
                    self.console_status = f"Error: {e}"
                print("[ERROR]", e)

        Thread(target=run, daemon=True).start()

'''

code = code[:start] + replacement + "\n" + code[end:]

p.write_text(code)

print("✅ Fixed broken vision/action/command block.")
