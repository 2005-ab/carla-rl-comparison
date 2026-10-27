"""Pygame presentation dashboard, also usable with SDL's dummy video driver."""
import math
import numpy as np
import pygame

BG = (12, 19, 29)
PANEL = (22, 33, 47)
WHITE = (230, 239, 246)
MUTED = (137, 159, 179)
CYAN = (56, 215, 218)
GOLD = (247, 189, 77)


class DrivingRenderer:
    def __init__(self, mode="human"):
        pygame.init()
        self.mode = mode
        self.surface = pygame.display.set_mode((1280, 800)) if mode == "human" else pygame.Surface((1280, 800))
        if mode == "human":
            pygame.display.set_caption("DRIVE LAB | PPO / SAC / TD3")
        self.font = pygame.font.SysFont("segoeui", 17)
        self.small = pygame.font.SysFont("consolas", 14)
        self.big = pygame.font.SysFont("segoeui", 32, bold=True)
        self.title = pygame.font.SysFont("segoeui", 21, bold=True)
        self.label = "BASELINE / hand-coded controller"
        self.paused = False
        self.comparison = []

    def text(self, value, pos, color=WHITE, font=None):
        self.surface.blit((font or self.font).render(str(value), True, color), pos)

    def panel(self, rect):
        pygame.draw.rect(self.surface, PANEL, rect, border_radius=12)

    def car(self, pos, yaw, color):
        sprite = pygame.Surface((48, 24), pygame.SRCALPHA)
        pygame.draw.rect(sprite, color, (1, 2, 46, 20), border_radius=6)
        pygame.draw.rect(sprite, (24, 45, 63), (26, 4, 10, 16), border_radius=3)
        pygame.draw.rect(sprite, (35, 57, 71), (9, 5, 9, 14), border_radius=2)
        pygame.draw.rect(sprite, (255, 248, 210), (43, 4, 3, 5))
        pygame.draw.rect(sprite, (255, 248, 210), (43, 15, 3, 5))
        rotated = pygame.transform.rotate(sprite, -math.degrees(yaw))
        self.surface.blit(rotated, rotated.get_rect(center=pos))

    def graph(self, values, rect, color, minimum=None, maximum=None):
        x, y, w, h = rect
        pygame.draw.line(self.surface, (55, 73, 89), (x, y+h), (x+w, y+h))
        if len(values) < 2:
            return
        values = values[-180:]
        lo = min(values) if minimum is None else minimum
        hi = max(values) if maximum is None else maximum
        points = [(x+i*w/(len(values)-1), y+h-h*(v-lo)/max(hi-lo, .001)) for i, v in enumerate(values)]
        pygame.draw.lines(self.surface, color, False, points, 2)

    def draw(self, env):
        self.surface.fill(BG)
        self.text("DRIVE LAB", (26, 15), font=self.big)
        self.text("CONTINUOUS CONTROL  /  LOCAL 2D SIMULATOR", (250, 30), MUTED, self.small)
        self.text(self.label, (26, 62), CYAN, self.title)
        self.text(f"{env.active_scenario.upper()}   |   seed-controlled scenarios   |   10 Hz physics", (665, 65), MUTED, self.small)
        self.panel((20, 105, 850, 450))
        # A forward-following view with a fixed world scale. Y points down on screen.
        viewport = pygame.Rect(22, 107, 846, 446)
        self.surface.set_clip(viewport)
        scale = 10
        def world(x, y):
            return (int(210 + (x-env.x)*scale), int(330 + (y-env.y)*scale))
        for gx in range(int(env.x)-30, int(env.x)+80, 10):
            pygame.draw.line(self.surface, (28, 47, 53), world(gx, env.y-30), world(gx, env.y+30))
        xs = np.linspace(env.x-25, env.x+72, 220)
        half = env.road_half_width
        upper = [world(x, float(env.road_center_y(x))-half) for x in xs]
        lower = [world(x, float(env.road_center_y(x))+half) for x in xs]
        pygame.draw.polygon(self.surface, (53, 66, 77), upper+lower[::-1])
        for offset in (-half, half):
            pygame.draw.lines(self.surface, (195, 206, 210), False, [world(x, float(env.road_center_y(x))+offset) for x in xs], 2)
        if env.is_two_lane:
            for offset in (-.12,.12):
                pygame.draw.lines(self.surface,GOLD,False,[world(x,float(env.road_center_y(x))+offset) for x in xs],1)
        else:
            for offset in (-2.5, 2.5):
                pygame.draw.lines(self.surface, (88, 108, 118), False, [world(x, float(env.road_y(x))+offset) for x in xs], 1)
            for start in np.arange(math.floor((env.x-25)/6)*6, env.x+72, 6):
                pygame.draw.line(self.surface, GOLD, world(start, float(env.road_y(start))), world(start+3, float(env.road_y(start+3))), 2)
        if len(env.trail)>1:
            pygame.draw.lines(self.surface, (40, 152, 161), False, [world(*p) for p in env.trail[-250:]], 2)
        goal = env.config.route.route_length_m
        pygame.draw.line(self.surface, CYAN, world(goal, float(env.road_center_y(goal))-half), world(goal, float(env.road_center_y(goal))+(0 if env.is_two_lane else half)), 5)
        for x, lat, speed in env.traffic:
            self.car(world(x, float(env.road_y(x))+lat), env.traffic_yaw((x,lat,speed)), (235,113,100) if speed<0 else GOLD)
        self.car(world(env.x, env.y), env.yaw, CYAN)
        car = env.nearest_traffic()
        if car and car[0]-env.x < 40:
            pygame.draw.line(self.surface, (184, 134, 66), world(env.x+2, env.y), world(car[0]-2, float(env.road_y(car[0]))), 1)
            self.text(f"{max(0, car[0]-env.x-4.4):.1f} m", (270, 280), GOLD, self.small)
        self.surface.set_clip(None)
        self.text("LIVE WORLD", (37, 121), MUTED, self.small)
        self.text("KEEP LEFT | Red: oncoming | Gold: lead traffic" if env.is_two_lane else "Cyan: ego vehicle   Gold: traffic   Teal: driven trajectory", (37, 525), MUTED, self.small)
        if env.is_two_lane:
            oncoming = env.nearest_oncoming()
            if oncoming:
                self.text(f"ONCOMING {max(0,oncoming[0]-env.x-4.4):.0f} m / {-oncoming[2]*3.6:.0f} km/h",(37,151),GOLD,self.small)
            if env.last_info["wrong_lane"]:
                self.text("WRONG LANE - MOVE LEFT",(37,178),(255,129,111),self.title)
        if env.done or self.paused:
            self.text("PAUSED" if self.paused else env.last_info["termination_reason"].upper(), (330, 175), GOLD, self.big)
        self.panel((890, 105, 370, 450))
        self.text("VEHICLE TELEMETRY", (910, 121), MUTED, self.small)
        self.text(f"{env.speed*3.6:05.1f}", (910, 147), CYAN, self.big)
        self.text("km/h", (1020, 164), MUTED)
        self.text(f"Target {env.config.route.target_speed_kmh:.0f}", (1130, 163), MUTED)
        lateral, heading = env.geometry()
        rows = [("Route complete", f"{env.last_info['route_completion_pct']:.1f}%"),
                ("Lane deviation", f"{lateral:+.2f} m"), ("Heading error", f"{math.degrees(heading):+.1f} deg"),
                ("Episode return", f"{env.total_reward:+.1f}"), ("Simulation time", f"{env.step_count*.1:.1f} s")]
        for i, (label, value) in enumerate(rows):
            self.text(label, (910, 212+i*31), MUTED)
            self.text(value, (1135, 212+i*31))
        progress = env.last_info["route_completion_pct"]/100
        pygame.draw.rect(self.surface, (44, 61, 76), (910, 383, 325, 6), border_radius=3)
        pygame.draw.rect(self.surface, CYAN, (910, 383, int(325*progress), 6), border_radius=3)
        for i, (name, val, color) in enumerate(zip(("STEER", "THROTTLE", "BRAKE"), env.previous_action, (CYAN, (116, 223, 154), GOLD))):
            yy = 411+i*40
            self.text(name, (910, yy), MUTED, self.small)
            self.text(f"{val:+.2f}", (1180, yy), color, self.small)
            pygame.draw.rect(self.surface, (44, 61, 76), (1010, yy+5, 150, 8), border_radius=3)
            pygame.draw.rect(self.surface, color, (1010, yy+5, int(150*abs(val)), 8), border_radius=3)
        self.panel((20, 573, 410, 151))
        self.text("SPEED  /  last 18 seconds", (37, 585), MUTED, self.small)
        self.graph(env.speed_history, (38, 615, 374, 83), CYAN, 0, 55)
        self.panel((446, 573, 424, 151))
        self.text("STEP REWARD  /  last 18 seconds", (463, 585), MUTED, self.small)
        self.graph(env.reward_history, (464, 615, 387, 83), GOLD)
        self.panel((890, 573, 370, 151))
        self.text("EVALUATION  /  held-out seeds", (907, 585), MUTED, self.small)
        if self.comparison:
            self.text("POLICY    RETURN   GOAL   ROUTE", (908, 614), MUTED, self.small)
            for i, row in enumerate(self.comparison[:4]):
                self.text(f"{row['algorithm'][:8]:8} {float(row['mean_reward']):6.0f} {float(row['goal_success_rate_pct']):5.0f}% {float(row['route_completion_pct_mean']):5.0f}%", (908, 637+i*19), font=self.small)
        else:
            self.text("Run train_local.py for measured results", (907, 631), MUTED, self.small)
        self.text("1 PPO  2 SAC  3 TD3  B Baseline  M Manual  R Reset  T Scenario  V 2D/3D  SPACE Pause  +/- Speed  ESC Exit", (26, 744), WHITE, self.small)
        self.text("Local prototype: bicycle dynamics, numeric state. CARLA integration is a separate backend; policies require retraining.", (26, 773), MUTED, self.small)
        if self.mode == "human":
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface), (1, 0, 2))

    def close(self):
        pygame.display.quit()
        pygame.quit()
