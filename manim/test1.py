from manim import *

import subprocess


class Test(Scene):
    def construct(self):
        c = Circle(2, color=RED, fill_opacity=0.1)
        self.play(DrawBorderThenFill(c), run_time=1)
        self.wait(1.5)


# manim -pql test1.py Test
