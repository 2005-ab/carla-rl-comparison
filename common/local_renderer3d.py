"""Small CPU 3D renderer: world-space meshes, perspective camera, clipping and fog.

Pygame draws projected polygons; no OpenGL driver or external assets are needed.
This changes visualization only. LocalDrivingEnv still supplies flat-road dynamics.
"""
import math
import numpy as np
import pygame
from common.local_renderer import CYAN, GOLD, WHITE, MUTED


def clip_near(vertices, near=0.35):
    """Clip a camera-space polygon against the near plane before perspective divide."""
    output = []
    for i, current in enumerate(vertices):
        previous = vertices[i-1]
        inside, was_inside = current[2] >= near, previous[2] >= near
        if inside != was_inside:
            ratio = (near-previous[2]) / (current[2]-previous[2])
            output.append(previous + ratio*(current-previous))
        if inside:
            output.append(current)
    return np.asarray(output)


class DrivingRenderer3D:
    cameras = ("CHASE", "HOOD", "AERIAL")

    def __init__(self, mode="human"):
        pygame.init()
        self.mode = mode
        self.surface = pygame.display.set_mode((1280, 800)) if mode == "human" else pygame.Surface((1280, 800))
        if mode == "human":
            pygame.display.set_caption("DRIVE LAB 3D | PPO / SAC / TD3")
        self.font = pygame.font.SysFont("segoeui", 18)
        self.small = pygame.font.SysFont("consolas", 14)
        self.title = pygame.font.SysFont("segoeui", 23, bold=True)
        self.big = pygame.font.SysFont("segoeui", 44, bold=True)
        self.label = "BASELINE / hand-coded reference"
        self.paused, self.comparison = False, []
        self.camera_index = 0
        self.faces = []
        self._scenery_key = None
        self._scenery = []
        self.focal = 800
        self.sky = pygame.Surface((1280, 800))
        for y in range(800):
            t = min(1, y / 380)
            color = tuple(int(a+(b-a)*t) for a,b in zip((67, 126, 177), (197, 220, 225)))
            pygame.draw.line(self.sky, color, (0,y), (1280,y))

    def cycle_camera(self):
        self.camera_index = (self.camera_index+1) % len(self.cameras)

    def set_camera(self, env):
        direction = np.array([math.cos(env.yaw), math.sin(env.yaw), 0.0])
        origin = np.array([env.x, env.y, 0.0])
        camera = self.cameras[self.camera_index]
        if camera == "HOOD":
            self.eye = origin + direction*.8 + [0,0,1.55]
            target = origin + direction*30 + [0,0,1.1]
        elif camera == "AERIAL":
            self.eye = origin - direction*14 + [0,0,26]
            target = origin + direction*10
        else:
            self.eye = origin - direction*10 + [0,0,5.4]
            target = origin + direction*12 + [0,0,.7]
        forward = target - self.eye
        forward /= np.linalg.norm(forward)
        right = np.cross([0,0,1], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        self.basis = np.array([right, up, forward])

    def face(self, points, color):
        self.faces.append((points, color))

    def project_faces(self, faces=None):
        """Transform/project batches in NumPy; clip only polygons crossing the camera."""
        groups = {}
        for points,color in (self.faces if faces is None else faces):
            groups.setdefault(len(points),[]).append((points,color))
        projected_faces = []
        for group in groups.values():
            vertices = np.asarray([face[0] for face in group],dtype=float)
            colors = np.asarray([face[1] for face in group],dtype=float)
            camera = (vertices-self.eye) @ self.basis.T
            z = camera[:,:,2]
            visible = (z.max(axis=1)>=.35) & (z.min(axis=1)<220)
            camera, colors = camera[visible], colors[visible]
            if not len(camera):
                continue
            crosses = camera[:,:,2].min(axis=1)<.35
            clipped = [(clip_near(poly),color) for poly,color in zip(camera[crosses],colors[crosses])]
            camera,colors = camera[~crosses],colors[~crosses]
            if len(camera):
                depth = camera[:,:,2].mean(axis=1)
                points = camera[:,:,:2]/camera[:,:,2,None]*self.focal
                points[:,:,0] += 640
                points[:,:,1] = 368-points[:,:,1]
                visible = (points[:,:,0].max(axis=1)>=0)&(points[:,:,0].min(axis=1)<=1280)&(points[:,:,1].max(axis=1)>=84)&(points[:,:,1].min(axis=1)<=744)
                fog = np.clip((depth-45)/185,0,.83)[:,None]
                colors = (colors*(1-fog)+np.array([182,205,208])*fog).astype(int)
                projected_faces.extend(zip(depth[visible].tolist(),map(tuple,colors[visible].tolist()),points[visible].tolist()))
            for poly,color in clipped:
                if len(poly)<3:
                    continue
                depth = float(poly[:,2].mean())
                points = poly[:,:2]/poly[:,2,None]*self.focal
                points[:,0] += 640
                points[:,1] = 368-points[:,1]
                if points[:,0].max()<0 or points[:,0].min()>1280 or points[:,1].max()<84 or points[:,1].min()>744:
                    continue
                projected_faces.append((depth,tuple(color.astype(int)),points.tolist()))
        return projected_faces

    def box(self, center, size, color, yaw=0):
        x,y,z = center
        lx,ly,lz = (s/2 for s in size)
        co,si = math.cos(yaw), math.sin(yaw)
        vertices = np.array([(x+u*co-v*si, y+u*si+v*co, z+w)
                             for w in (-lz,lz) for u,v in ((-lx,-ly),(lx,-ly),(lx,ly),(-lx,ly))])
        for indices, light in [((0,1,5,4),.76), ((1,2,6,5),.9), ((2,3,7,6),.66), ((3,0,4,7),.82), ((4,5,6,7),1.1)]:
            self.face(vertices[list(indices)], tuple(min(255,int(c*light)) for c in color))

    def car(self, x, y, yaw, color, brake=0):
        co,si = math.cos(yaw),math.sin(yaw)
        def p(a,b,c):
            return (x+a*co-b*si, y+a*si+b*co, c)
        # Wheels are octagonal cylinders, with visible rotating hub spokes.
        for a in (-1.4,1.35):
            for b in (-.92,.92):
                rings = [[p(a+.36*math.cos(t),b+offset,.39+.36*math.sin(t))
                          for t in np.linspace(0,2*math.pi,9)[:-1]] for offset in (-.12,.12)]
                self.face(rings[0], (25,29,34))
                self.face(rings[1], (36,41,46))
                for k in range(8):
                    self.face([rings[0][k],rings[0][(k+1)%8],rings[1][(k+1)%8],rings[1][k]], (29,33,37))
                outside = b+(.125 if b>0 else -.125)
                self.face([p(a+.19*math.cos(t),outside,.39+.19*math.sin(t)) for t in np.linspace(0,2*math.pi,9)[:-1]], (153,166,177))
        self.box(p(0,0,.66), (4.4,1.8,.65), color, yaw)
        self.box(p(-.1,0,1.17), (2.35,1.53,.59), color, yaw)
        # Slanted windshield and rear glass extend to the roof.
        self.face([p(1.12,-.72,.99),p(1.12,.72,.99),p(.66,.64,1.52),p(.66,-.64,1.52)], (49,79,98))
        self.face([p(-1.35,.72,.99),p(-1.35,-.72,.99),p(-.92,-.64,1.52),p(-.92,.64,1.52)], (40,65,81))
        self.face([p(-.92,-.64,1.52),p(.66,-.64,1.52),p(.66,.64,1.52),p(-.92,.64,1.52)], tuple(min(255,int(c*1.14)) for c in color))
        for side in (-1,1):
            yy = side*.775
            self.face([p(-1.13,yy,1.03),p(.9,yy,1.03),p(.56,yy,1.43),p(-.82,yy,1.43)], (42,71,90))
            self.box(p(-.15,yy,1.24), (.08,.035,.4), color, yaw)
            self.box(p(-2.215,side*.62,.76), (.035,.4,.19), (255,55,45) if brake>.05 else (159,35,42), yaw)
            self.box(p(2.215,side*.62,.77), (.035,.38,.18), (248,246,211), yaw)
        self.box(p(-2.23,0,.56), (.025,.47,.14), (230,234,221), yaw)
        self.box(p(-2.22,0,.37), (.04,1.65,.11), (36,43,51), yaw)

    def road_strip(self, env, x, end, left, right, height, color):
        self.face([(x,float(env.road_center_y(x))+left,height), (end,float(env.road_center_y(end))+left,height),
                   (end,float(env.road_center_y(end))+right,height), (x,float(env.road_center_y(x))+right,height)], color)

    def tree(self, x, y, height, variant):
        self.box((x,y,1.1), (.35,.35,2.2), (100,81,55))
        for base,radius in ((1.7,height*.33),(height*.46,height*.26)):
            points = [(x+radius*math.cos(t),y+radius*math.sin(t),base) for t in np.linspace(0,2*math.pi,7)[:-1]]
            tip = (x,y,height)
            for i in range(6):
                self.face([points[i],points[(i+1)%6],tip], (40+variant*3,100+i*5,65+i*2))

    def world(self, env):
        self.faces = []
        self.set_camera(env)
        start = math.floor((env.x-40)/4)*4
        stop = env.x+150
        key = (env.active_scenario,env.amplitude,env.phase,env.config.route.route_length_m,math.floor(env.x/16))
        if key != self._scenery_key:
            self.scenery(env,start,stop)
            self._scenery = self.faces.copy()
            self._scenery_key = key
        else:
            self.faces = self._scenery.copy()
        for idx,(x,lat,speed) in enumerate(env.traffic):
            if -25 < x-env.x < 160:
                color = (224,105,92) if speed<0 else [(243,178,66),(126,155,206)][idx%2]
                self.car(x,float(env.road_y(x))+lat,env.traffic_yaw((x,lat,speed)),color)
        if self.cameras[self.camera_index] != "HOOD":
            self.car(env.x,env.y,env.yaw,(42,190,199),env.previous_action[2])
        self.surface.blit(self.sky,(0,0))
        forward_ground = np.array([math.cos(env.yaw),math.sin(env.yaw),0.0]) @ self.basis.T
        horizon = int(368-self.focal*forward_ground[1]/max(.01,forward_ground[2]))
        ground_start = max(84,min(744,horizon))
        pygame.draw.rect(self.surface,(108,141,104),(0,ground_start,1280,744-ground_start))
        if horizon>100:
            pygame.draw.circle(self.surface,(244,235,192),(940,max(120,horizon-85)),30)
            ridge = [(0,horizon+4),(140,horizon-55),(245,horizon-8),(350,horizon-44),(510,horizon+2),
                     (650,horizon-30),(800,horizon+2),(945,horizon-50),(1100,horizon-7),(1210,horizon-44),(1280,horizon-8),(1280,horizon+10),(0,horizon+10)]
            pygame.draw.polygon(self.surface,(144,173,172),ridge)
        self.surface.set_clip(pygame.Rect(0,84,1280,660))
        # Paint the flat road as its own layer. Sorting coplanar lane markings
        # together with asphalt can create seams with a painter's algorithm.
        for depth,color,points in self.project_faces(self._road_faces):
            pygame.draw.polygon(self.surface,color,points)
        for depth,color,points in sorted(self.project_faces(),key=lambda f:f[0],reverse=True):
            pygame.draw.polygon(self.surface,color,points)
        self.surface.set_clip(None)

    def scenery(self, env, start, stop):
        half = env.road_half_width
        for x in range(start,int(stop),2):
            self.road_strip(env,x,x+2,-half-.7,half+.7,-.025,(158,153,133))
        for x in range(start,int(stop),2):
            self.road_strip(env,x,x+2,-half,half,0,(61,70,78))
        for x in range(start,int(stop),2):
            for offset in (-half+.18,half-.28):
                self.road_strip(env,x,x+2,offset,offset+.10,.008,(228,229,212))
            if env.is_two_lane:
                for offset in (-.16,.08):
                    self.road_strip(env,x,x+2,offset,offset+.08,.012,(242,198,89))
            elif (x//2)%3 != 2:
                self.road_strip(env,x,x+2,-.055,.055,.012,(242,198,89))
        if env.is_two_lane:
            for x in range(math.floor(start/24)*24,int(stop),24):
                for lane,direction in ((-1.8,1),(1.8,-1)):
                    # Painted arrows distinguish the two travel directions.
                    points = [(-1.5,-.10),(.3,-.10),(.3,-.4),(1.5,0),(.3,.4),(.3,.10),(-1.5,.10)]
                    self.face([(x+u*direction,float(env.road_center_y(x+u*direction))+lane+v,.018) for u,v in points],(219,225,215))
        self._road_faces = self.faces.copy()
        self.faces = []
        for x in range(math.floor(start/16)*16,int(stop),16):
            for side in (-1,1):
                variation = (x*17+side*13)%9
                y = float(env.road_center_y(x))+side*(10+variation*.6)
                self.tree(x,y,5+variation*.35,variation)
                # Road delineators and their white reflectors.
                py = float(env.road_center_y(x))+side*(half+.5)
                self.box((x,py,.52),(.16,.16,1.04),(222,225,208))
                self.box((x,py,.8),(.19,.19,.17),(239,183,66))
        for x in range(math.floor(start/32)*32,int(stop),32):
            side = 1 if (x//32)%2 else -1
            y = float(env.road_center_y(x))+side*23
            height = 5+(x*7)%10
            color = [(159,157,141),(173,163,143),(140,160,165)][(x//32)%3]
            self.box((x,y,height/2),(9,8,height),color)
            self.box((x,y,height+.2),(9.6,8.6,.4),(80,96,108))
            for level in range(2,int(height),3):
                for xx in (-2.7,0,2.7):
                    self.box((x+xx,y-side*4.015,level),(1.3,.025,1.2),(73,108,127))
            # Street lights on the outer shoulder.
            ly = float(env.road_center_y(x))+side*(half+1.5)
            self.box((x,ly,3.2),(.14,.14,6.4),(101,112,118))
            self.box((x,ly-side*.8,6.4),(.15,1.7,.12),(112,123,127))
            self.box((x,ly-side*1.5,6.35),(.5,.75,.13),(241,229,177))
        goal = env.config.route.route_length_m
        finish_width = half if env.is_two_lane else half*2
        for i in range(math.ceil(finish_width/.5)):
            offset = -half+i*.5
            for j in range(2):
                self.road_strip(env,goal+j*.5,goal+(j+1)*.5,offset,min(offset+.5,-half+finish_width),.02,(235,239,229) if (i+j)%2 else (36,45,54))
        # Roadside finish markers (decorative, no hidden collision obstacles).
        for side in (-1,1):
            y = float(env.road_center_y(goal))+side*(half+1.3)
            self.box((goal,y,2),(.12,.12,4),(61,91,101))
            self.box((goal,y,3.5),(.18,1.1,.8),(53,191,179))

    def text(self, text, pos, color=WHITE, font=None):
        self.surface.blit((font or self.font).render(str(text),True,color),pos)

    def panel(self, rect, alpha=225):
        overlay = pygame.Surface((rect[2],rect[3]),pygame.SRCALPHA)
        pygame.draw.rect(overlay,(14,25,38,alpha),overlay.get_rect(),border_radius=12)
        self.surface.blit(overlay,rect[:2])

    def hud(self, env):
        self.panel((0,0,1280,84),248)
        self.text("DRIVE LAB",(22,10),font=self.title)
        self.text("3D",(164,10),CYAN,self.title)
        self.text(self.label,(230,13),CYAN)
        self.text(f"{env.active_scenario.upper()}   /   {self.cameras[self.camera_index]} CAMERA" + ("   /   KEEP LEFT" if env.is_two_lane else ""),(22,49),MUTED,self.small)
        self.text("3D VIEW  /  BICYCLE DYNAMICS  /  LOCAL POLICY",(824,51),MUTED,self.small)
        # Telemetry cards leave the central driving view clear.
        self.panel((1015,102,245,330))
        self.text("VEHICLE TELEMETRY",(1032,119),MUTED,self.small)
        self.text(f"{env.speed*3.6:04.1f}",(1030,139),font=self.big)
        self.text("km/h",(1157,169),MUTED)
        lateral,heading = env.geometry()
        for idx,(name,value) in enumerate([
            ("Lane error",f"{lateral:+.2f} m"),("Heading",f"{math.degrees(heading):+.1f} deg"),
            ("Return",f"{env.total_reward:+.1f}"),("Time",f"{env.step_count*.1:.1f} s")]):
            self.text(name,(1032,211+idx*31),MUTED)
            self.text(value,(1157,211+idx*31))
        self.text("ROUTE COMPLETION",(1032,350),MUTED,self.small)
        progress = env.last_info["route_completion_pct"]
        pygame.draw.rect(self.surface,(48,65,77),(1032,379,207,7),border_radius=3)
        pygame.draw.rect(self.surface,CYAN,(1032,379,int(207*progress/100),7),border_radius=3)
        self.text(f"{progress:.1f}%    {env.last_info['distance_to_goal_m']:.0f} m remaining",(1032,398),font=self.small)
        self.panel((20,102,240,178))
        self.text("ROUTE MAP",(36,115),MUTED,self.small)
        length = env.config.route.route_length_m
        def mp(x,y):
            return (int(40+np.clip(x/length,0,1)*195),int(193+y*3))
        route = [mp(x,float(env.road_center_y(x))) for x in np.linspace(0,length,100)]
        pygame.draw.lines(self.surface,(92,115,129),False,route,7)
        pygame.draw.lines(self.surface,(215,221,212),False,route,1)
        for x,lat,speed in env.traffic:
            if 0<=x<=length:
                pygame.draw.circle(self.surface,(244,121,106) if speed<0 else GOLD,mp(x,float(env.road_y(x))+lat),4)
        pygame.draw.circle(self.surface,CYAN,mp(env.x,env.y),6)
        self.text("START",(34,247),MUTED,self.small)
        self.text("GOAL",(201,247),MUTED,self.small)
        car = env.nearest_traffic()
        if car and car[0]-env.x < 45:
            self.panel((480,101,320,37),208)
            gap = max(0,car[0]-env.x-4.4)
            self.text(f"TRAFFIC AHEAD  {gap:4.1f} m  /  {car[2]*3.6:.0f} km/h",(495,110),GOLD,self.small)
        if env.is_two_lane:
            oncoming = env.nearest_oncoming()
            self.panel((456,147,368,37),225)
            if env.last_info["wrong_lane"]:
                self.text("WRONG LANE - MOVE LEFT",(472,156),(255,129,111),self.small)
            elif oncoming:
                self.text(f"ONCOMING {max(0,oncoming[0]-env.x-4.4):4.0f} m / {-oncoming[2]*3.6:.0f} km/h - KEEP LEFT",(472,156),(255,174,137),self.small)
            else:
                self.text("TWO-WAY ROAD / KEEP LEFT",(472,156),CYAN,self.small)
        self.panel((20,623,340,104))
        for i,(name,val,color) in enumerate(zip(("STEER","THROTTLE","BRAKE"),env.previous_action,(CYAN,(122,223,157),GOLD))):
            yy = 638+i*28
            self.text(name,(36,yy),MUTED,self.small)
            pygame.draw.rect(self.surface,(47,65,77),(130,yy+4,146,8),border_radius=3)
            pygame.draw.rect(self.surface,color,(130,yy+4,int(146*abs(val)),8),border_radius=3)
            self.text(f"{val:+.2f}",(292,yy),color,self.small)
        self.panel((1015,470,245,257))
        self.text("HELD-OUT EVALUATION",(1032,487),MUTED,self.small)
        self.text("POLICY   RETURN  GOAL",(1032,519),MUTED,self.small)
        for i,row in enumerate(self.comparison[:4]):
            self.text(f"{row['algorithm'][:8]:8} {float(row['mean_reward']):5.0f} {float(row['goal_success_rate_pct']):4.0f}%",(1032,546+i*25),font=self.small)
        if not self.comparison:
            self.text("Train models for results",(1032,550),font=self.small)
        self.text("Two-lane challenge results" if env.is_two_lane else "Single seed / local 2D task",(1032,673),MUTED,self.small)
        self.text("3D scenery is decorative",(1032,698),MUTED,self.small)
        if self.paused or env.done:
            self.panel((395,320,490,102),240)
            label = "PAUSED" if self.paused else env.last_info["termination_reason"].upper()
            self.text(label,(420,337),GOLD,self.title)
            self.text("SPACE resume  /  R restart  /  1 2 3 switch policy",(420,382),MUTED,self.small)
        self.panel((0,744,1280,56),248)
        self.text("1 PPO   2 SAC   3 TD3   B Baseline   M Manual   R Reset   T Scenario   C Camera   V 2D/3D",(22,753),font=self.small)
        self.text("WASD / arrows: drive   SPACE: pause   +/-: speed   N: new seed   ESC: exit",(22,779),MUTED,self.small)

    def draw(self, env):
        self.world(env)
        self.hud(env)
        if self.mode == "human":
            pygame.display.flip()
            return None
        return np.transpose(pygame.surfarray.array3d(self.surface),(1,0,2))

    def close(self):
        pygame.display.quit()
        pygame.quit()
