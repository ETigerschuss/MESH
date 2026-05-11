from scipy.ndimage import gaussian_filter
import numpy as np
import time
import mmap
import pandas as pd
import math
import pygame
from jax import jit
import jax.numpy as jnp


class SuperBowl:

    def __init__(
        self,
        screen_type="Flat",
        xfov=360,
        yfov=90,
        xres_scale=2,
        yres_scale=2,
        upside_down=False,
        fov_ele=10,
        rot_offset=(0, 0, 0),
        margin=[0, 0],
    ):

        self.xres_scale = xres_scale
        self.yres_scale = yres_scale
        self.xdim = xfov * self.xres_scale
        self.ydim = yfov * self.yres_scale

        pygame.init()
        display_info = pygame.display.get_desktop_sizes()
        display_id = len(display_info) - 1
        screen_res = display_info[display_id]
        self.screen = pygame.display.set_mode(
            screen_res, pygame.FULLSCREEN | pygame.SCALED, display=display_id
        )

        unchanged_image = lambda pic: pic
        self.upright_image = (
            (lambda pic: self.jit_flip_vertically(pic)) if upside_down else unchanged_image
        )

        self.projected_image = 0
        self.flatten_image = (
            lambda pic: pygame.transform.scale(pygame.surfarray.make_surface(pic), screen_res)
        )

        if screen_type == "Flat":
            self.projected_image = self.flatten_image
        else:
            self.projector = BowlProjection(
                xfov,
                yfov,
                self.xres_scale,
                self.yres_scale,
                fov_ele,
                is_360=screen_type == "FullBowl",
                screen_res=screen_res,
                rot_offset=rot_offset,
                margin=margin,
            )
            self.projected_image = lambda pic: pygame.surfarray.make_surface(
                self.projector.flat_to_bowl(pic)
            )

    def display_image(self, pic, distortion_fun):
        pic = self.upright_image(pic)
        if pic.ndim == 2:
            pic = jit_grey_to_rgb(pic)
        array_surface = distortion_fun(pic)
        self.screen.blit(array_surface, (0, 0))
        pygame.display.flip()

    def superpose(self, layers):
        return jit_superpose(layers)

    def split_display_vertically(self, left_panel, right_panel, mid_line=0.5):
        mask = np.ones((self.xdim, self.ydim))
        mask[int(self.xdim * mid_line):, :] = 0
        return np.where(mask, right_panel, left_panel).astype(np.uint8)

    def split_display_vertically_binoRemoved_side(
        self, left_panel, right_panel, bino_width, lum_overlap, side, mid_line=0.5
    ):
        mask = np.ones((self.xdim, self.ydim))
        mask[int(round(self.xdim * mid_line)):, :] = 0

        frontal_width = int(round(bino_width / 360 * self.xdim))
        left_limit = int(round(mid_line * self.xdim - frontal_width / 2))
        right_limit = int(round(mid_line * self.xdim + frontal_width / 2))

        if side == 2:
            mask = 1 - mask
        mask = np.where(mask, right_panel, left_panel).astype(np.uint8)
        mask[left_limit:right_limit, :] = lum_overlap
        return mask

    def remove_bino(self, stimulus, binoLum, bino_width, mid_line=0.5):
        mask = np.zeros((self.xdim, self.ydim))
        frontal_width = int(bino_width / 360 * self.xdim)
        left_limit = int(mid_line * self.xdim - frontal_width / 2)
        right_limit = int(mid_line * self.xdim + frontal_width / 2)
        mask[left_limit:right_limit, :] = 1
        return np.where(mask, stimulus, binoLum).astype(np.uint8)

    def shift(self, pic, xshift=0, yshift=0):
        pic = jit_shift(pic, xshift * self.xres_scale, yshift * self.yres_scale)
        return pic

    def is_ESC_pressed(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def has_eye_recording_stopped(self):
        with mmap.mmap(-1, 1024, "EyeTracker") as mm:
            line = mm.readline().decode().strip()
            if "Recording_Stopped" in line:
                print("exit")
                return True
        return False

    def grey_screen(self, color):
        return np.ones([self.xdim, self.ydim], dtype=np.uint8) * color

    def picture(self, path, grey_scaled=False):
        img = pygame.image.load(path).convert()
        pic = pygame.surfarray.array3d(img)
        if grey_scaled:
            pic = (
                0.2989 * pic[:, :, 0]
                + 0.5870 * pic[:, :, 1]
                + 0.1140 * pic[:, :, 2]
            )
        return pic.astype(np.uint8)

    def grating_vertical(self, freq, color=0, color_b=255, offset=0, proportion=0.5):
        freq *= self.xres_scale
        x = jnp.arange(self.xdim) + (offset * self.xres_scale)
        pattern = (x % freq) < (freq * proportion)
        mask = jnp.tile(pattern[:, None], (1, self.ydim))
        pic = jnp.where(mask, color, color_b)
        return pic.astype(jnp.uint8)

    def grating_horizontal(self, freq, color=0, color_b=255, offset=0, proportion=0.5):
        freq *= self.yres_scale
        y = np.arange(self.ydim) + (offset * self.yres_scale)
        pattern = (y % freq) < (freq * proportion)
        mask = jnp.tile(pattern, (self.xdim, 1))
        pic = jnp.where(mask, color, color_b)
        return pic.astype(jnp.uint8)

    def grating_sine_vertical(self, amplitude, freq, offset=127.5, phase=0):
        f = 360 / freq
        t = np.linspace(0, np.pi * f * self.xres_scale, self.xdim)
        val = amplitude * np.sin(t + phase) + offset
        pic = np.outer(val, np.ones(self.ydim))
        return np.clip(pic, 0, 255).astype(np.uint8)

    def grating_sine_horizontal(self, amplitude, freq, offset=0):
        f = int(360 / freq)
        t = np.linspace(0, np.pi * f * self.xres_scale, self.ydim)
        val = amplitude * np.sin(t) + (offset * self.yres_scale)
        xs = np.ones(self.xdim)
        pic = np.outer(xs, val)
        return pic.astype(np.uint8)

    def bar_vertical(self, width, color=0, color_b=255, offset=0):
        width *= self.xres_scale
        offset = int((self.xdim / 2) + (offset * self.xres_scale) - (width / 2))
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_b
        pic[offset:(offset + width), :] = color
        return pic

    def bar_horizontal(self, width, color=0, color_b=255, offset=0):
        width *= self.yres_scale
        offset = int((self.xdim / 2) + (offset * self.yres_scale) - (width / 2))
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_b
        pic[:, offset:(offset + width)] = color
        return pic.astype(np.uint8)

    def rectangle(self, width, height, color=0, color_b=255, offset_x=0, offset_y=0):
        width *= self.xres_scale
        height *= self.yres_scale
        offset_x = int((self.xdim / 2) + (offset_x * self.xres_scale) - (width / 2))
        offset_y = int((self.ydim / 2) + (offset_y * self.yres_scale) - (height / 2))
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_b
        pic[offset_x:(offset_x + width), offset_y:(offset_y + height)] = color
        return pic

    def checker_bar_vertical(
        self, square_size=5, columns=3, color1=0, color2=254, color_b=100, offset=0
    ):
        square_sizex = square_size * self.xres_scale
        square_sizey = square_size * self.yres_scale

        rows = math.ceil(self.ydim / square_sizey)
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_b
        offset = int((self.xdim / 2) + (offset * self.xres_scale) - (columns * square_sizex / 2))
        for i in range(columns):
            square_posx = (i * square_sizex) + offset
            for j in range(rows):
                square_posy = j * square_sizey
                pic[
                    square_posx:(square_posx + square_sizey),
                    square_posy:(square_posy + square_sizey),
                ] = np.random.choice([color1, color2])
        return pic

    def grid(self, cell_size):
        pic = np.ones((self.ydim, self.xdim), dtype=np.uint8) * 255
        for x in range(0, self.xdim, cell_size * self.xres_scale):
            pic[x, :] = 0
        for y in range(0, self.ydim, cell_size * self.yres_scale):
            pic[:, y] = 0
        return pic

    def cross(self, xpos=0, ypos=0):
        pic = np.ones((self.ydim, self.xdim), dtype=np.uint8) * 255
        pic[int(xpos * self.yres_scale), int(ypos * self.yres_scale)] = 0
        return pic

    def gaussian_islands(self, island_size=15, threshold=0.5, contrast=1, seed=None):
        if seed is not None:
            np.random.seed(seed)
        noise = np.random.rand(*(self.xdim, self.ydim))
        smoothed_noise = gaussian_filter(noise, sigma=island_size)
        pattern = smoothed_noise > threshold
        i_min = (1 - contrast) / (1 + contrast)
        pattern = np.where(pattern, 1, i_min) * 255
        return pattern.astype(np.uint8)

    def checker_screen(self, pixel_size, color1, color2):
        pixel_size *= self.xres_scale
        squares_per_row = int(self.xdim / pixel_size)
        squares_per_col = int(self.ydim / pixel_size)
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8)
        for i in range(squares_per_row):
            for j in range(squares_per_col):
                offsetx = i * pixel_size
                offsety = j * pixel_size
                pic[offsetx:(offsetx + pixel_size), offsety:(offsety + pixel_size)] = np.random.choice(
                    [color1, color2]
                )
        return pic

    def disc(self, radius, color_disc=0, color_bg=255, center=None):
        if center is None:
            center = (int(self.xdim / 2), int(self.ydim / 2))
        else:
            center = np.asarray([center[0] * self.xres_scale, center[1] * self.yres_scale])

        x_grid, y_grid = np.mgrid[:self.xdim, :self.ydim]
        dist_from_center = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_bg
        pic[dist_from_center < (radius * self.xres_scale)] = color_disc
        return pic

    def distant_disc(self, radius, distance, color_disc=0, color_bg=255, center=None):
        if distance <= 0:
            return np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_disc

        if center is None:
            center = (int(self.xdim / 2), int(self.ydim / 2))
        else:
            center = np.asarray([center[0] * self.xres_scale, center[1] * self.yres_scale])

        x_grid, y_grid = np.mgrid[:self.xdim, :self.ydim]
        dist_from_center = np.sqrt((x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2)
        angular_radius_size = np.rad2deg(np.arctan(float(radius * self.xres_scale) / distance))
        pic = np.ones([self.xdim, self.ydim], dtype=np.uint8) * color_bg
        pic[dist_from_center < angular_radius_size] = color_disc
        return pic

    def grating_wave_2d(
        self,
        wavelength,
        speed_dps,
        direction_deg,
        t,
        wave_type="square",
        mean_lum=127.5,
        contrast=1.0,
        phase_offset_deg=0.0,
    ):
        x = (np.arange(self.xdim) - self.xdim / 2.0) / float(self.xres_scale)
        y = (np.arange(self.ydim) - self.ydim / 2.0) / float(self.yres_scale)
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        theta = np.deg2rad(direction_deg)
        spatial_pos = x_grid * np.cos(theta) + y_grid * np.sin(theta)

        wl = max(float(wavelength), 1e-6)
        phase_cycles = (spatial_pos / wl) - ((speed_dps * t + phase_offset_deg) / wl)

        wave_type = str(wave_type).strip().lower()
        mean_lum = float(mean_lum)
        contrast = float(np.clip(contrast, 0.0, 1.0))

        if wave_type == "sine":
            img = mean_lum + (mean_lum * contrast) * np.sin(2.0 * np.pi * phase_cycles)
        else:
            low = mean_lum * (1.0 - contrast)
            high = mean_lum * (1.0 + contrast)
            mask = np.mod(phase_cycles, 1.0) < 0.5
            img = np.where(mask, low, high)

        return np.clip(img, 0, 255).astype(np.uint8)

    def curtain_close(self, stimulus, progress, close_value=0, axis_deg=0.0, closure_deg=None):
        """Close a uniform-colour curtain over a stimulus from both edges inward.

        Parameters
        ----------
        stimulus    : 2-D uint8 array – the underlying pattern.
        progress    : float 0–1 – fraction of the timer elapsed (provided by loop_scenes).
        close_value : int 0–255 – luminance of the curtain:  255 = bright/ON,  0 = dark/OFF.
        axis_deg    : float – direction axis along which the curtain closes (visual degrees).
        closure_deg : float or None
            Each curtain panel travels this many visual degrees inward.
            None  → full closure (curtain goes all the way to the centre).
            Set to ``wavelength_deg`` (e.g. 30) so that one timer period covers
            exactly one spatial period (e.g. 30 deg / 30 deg·s⁻¹ = 1 s).
        """
        p = float(np.clip(progress, 0.0, 1.0))
        close_value = int(np.clip(close_value, 0, 255))

        x = (np.arange(self.xdim) - self.xdim / 2.0) / float(self.xres_scale)
        y = (np.arange(self.ydim) - self.ydim / 2.0) / float(self.yres_scale)
        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")

        theta = np.deg2rad(axis_deg)
        coord = x_grid * np.cos(theta) + y_grid * np.sin(theta)

        max_abs = float(np.max(np.abs(coord)))

        if closure_deg is None:
            # Full closure: curtain travels from edge all the way to the centre.
            remaining_half_width = (1.0 - p) * max_abs
        else:
            # Partial closure: each panel moves closure_deg degrees inward.
            # At p=0 → no curtain.  At p=1 → each panel has moved closure_deg inward.
            remaining_half_width = max(0.0, max_abs - p * float(closure_deg))

        keep_center = np.abs(coord) <= remaining_half_width
        return np.where(keep_center, stimulus, close_value).astype(np.uint8)

    def SQWEdges(self, progress, axis_deg=0.0, close_value=0,
                 wavelength=30.0, mean_lum=127.5, contrast=1.0):
        """Square-Wave Edges (SQWEdges) stimulus.

        Presents a full, *static* square-wave grating and simultaneously closes
        a bilateral uniform-colour curtain inward along ``axis_deg``.
        Each curtain panel moves exactly ``wavelength`` degrees, so the total
        stimulus duration equals  wavelength / speed_dps  (e.g. 30 / 30 = 1 s).

        At progress = 0   the screen shows the complete square-wave grating.
        At progress = 1   each curtain panel has covered one full wavelength.

        Parameters
        ----------
        progress    : float 0–1    provided by loop_scenes (p variable)
        axis_deg    : float [deg]  grating motion axis = curtain closure axis
        close_value : int 0–255   255 = ON (bright curtain),  0 = OFF (dark curtain)
        wavelength  : float [deg]  spatial wavelength of the underlying grating
        mean_lum    : float 0–255  mean luminance
        contrast    : float 0–1    Michelson contrast
        """
        grating = self.grating_wave_2d(
            wavelength=wavelength, speed_dps=0, direction_deg=axis_deg,
            t=0, wave_type="square", mean_lum=mean_lum, contrast=contrast,
        )
        return self.curtain_close(
            grating, progress,
            close_value=close_value, axis_deg=axis_deg, closure_deg=wavelength,
        )

    def SWEdges(self, progress, axis_deg=0.0, close_value=0,
                wavelength=30.0, mean_lum=127.5, contrast=1.0):
        """Sine-Wave Edges (SWEdges) stimulus.

        Identical to SQWEdges but the underlying pattern is a *sine wave*
        instead of a square wave.  Useful when a smooth luminance profile
        is preferred over the binary square-wave contrast.

        Parameters
        ----------
        progress    : float 0–1
        axis_deg    : float [deg]
        close_value : int 0–255   255 = ON (bright),  0 = OFF (dark)
        wavelength  : float [deg]
        mean_lum    : float 0–255
        contrast    : float 0–1
        """
        grating = self.grating_wave_2d(
            wavelength=wavelength, speed_dps=0, direction_deg=axis_deg,
            t=0, wave_type="sine", mean_lum=mean_lum, contrast=contrast,
        )
        return self.curtain_close(
            grating, progress,
            close_value=close_value, axis_deg=axis_deg, closure_deg=wavelength,
        )

    def move_with_FicTrac(
        self,
        texture,
        fictracKey,
        updateFunc,
        MMapName="2305",
        duration=0.0,
        EyeTracker_linked=True,
    ):
        old_frame = texture
        logs = []
        cumulative_value = 0.0
        fictrac_connected = True

        fictrack_mapping = {
            "right_shift": 6,
            "forward_shift": 7,
            "yaw_shift": 8,
            "right_pos": 12,
            "forward_pos": 13,
            "yaw_pos": 14,
            "speed": 19,
        }

        if fictracKey not in fictrack_mapping:
            print(f"ERROR: {fictracKey} is not valid a valid key")
            print(f"Here is the list of valid keys: {fictrack_mapping.keys()}")
            return

        keyIndex = fictrack_mapping[fictracKey]
        run_start_time = time.time()

        no_external_termination = (
            (lambda: not (self.is_ESC_pressed() | self.has_eye_recording_stopped()))
            if EyeTracker_linked
            else (lambda: not self.is_ESC_pressed())
        )
        no_timer_termination = lambda: True if duration == 0.00 else time.time() < run_start_time + duration

        frame_counter = 0
        while no_external_termination() and no_timer_termination():
            frame_counter += 1
            pic = old_frame
            now = time.time()

            try:
                caught_fictrac = False
                with mmap.mmap(-1, 1024, MMapName) as mm:
                    line = mm.readline().decode().strip()
                    toks = line.split(", ")

                    if (len(toks) > 24) and (toks[0] == "FT"):
                        caught_fictrac = True
                        instant_value = math.degrees(float(toks[keyIndex]))
                        cumulative_value += instant_value
                        pic = updateFunc(texture, instant_value, cumulative_value)
                        logs.append(
                            {
                                "Absolute_Time_ms": now * 1000,
                                f"Instant_{fictracKey}": instant_value,
                                f"Cumulative_{fictracKey}": cumulative_value,
                            }
                        )

            except Exception as exc:
                print(f"Error: {str(exc)}")

            if caught_fictrac:
                if not fictrac_connected:
                    print(f"Fictrac (MMapName: {MMapName}) reconnected")
                    fictrac_connected = True
            elif fictrac_connected:
                fictrac_connected = False
                print(f"Fictrac (MMapName: {MMapName}) is not connected")

            old_frame = pic
            self.display_image(pic, self.projected_image)

        print("mean fps " + str(frame_counter / (time.time() - run_start_time)))
        return pd.DataFrame(logs)

    class scene:
        def __init__(
            self,
            name=None,
            layers=None,
            layers_brk=None,
            start_delay=None,
            starting_screen=None,
            starting_screen_flat=None,
            timer=None,
            timer_iterations=None,
            timer_break=None,
            update=None,
            timer_update=None,
            seed=None,
            starting_lambda=None,
            stiCond_save=None,
            tF_save=None,
            wl_save=None,
            lum_save=None,
            eye_save=None,
            dir_save=None,
            pha_save=None,
            brk_save=None,
        ):
            self.name = name
            self.layers = layers
            self.layers_brk = layers_brk
            self.start_delay = start_delay
            self.starting_screen = starting_screen
            self.seed = seed
            self.timer = timer
            self.timer_iterations = timer_iterations
            self.timer_break = timer_break
            self.update = update
            self.timer_update = timer_update
            self.starting_screen_flat = starting_screen_flat
            self.starting_lambda = starting_lambda
            self.tF_save = tF_save
            self.stiCond_save = stiCond_save
            self.wl_save = wl_save
            self.lum_save = lum_save
            self.eye_save = eye_save
            self.dir_save = dir_save
            self.pha_save = pha_save
            self.brk_save = brk_save

    def loop_scenes(
        self,
        scenes,
        scene_timers=1.0,
        timer_iterations=1,
        timer_breaks=0.0,
        start_delays=0.0,
        starting_screens=None,
        starting_screens_flat=False,
        iteration=1,
        random_order=False,
        EyeTracker_linked=True,
    ):

        scenes_per_loop = len(scenes)
        scene_order = np.arange(scenes_per_loop)
        scene_defaultImage = self.grey_screen(0)
        current_bouncing_dir = 0
        current_scene_idx = 0
        current_loop = 0
        current_scene = 0
        current_timer = 0
        loop_start = 0.0
        scene_break_start = 0.0
        scene_start = 0.0
        timer_start = 0.0
        in_scene_break = False
        in_timer_break = False
        if starting_screens is None:
            starting_screens = self.grey_screen(255 / 2)
        now = time.time()
        timer_over = False
        logs = []

        def createLogRow(
            event,
            id,
            start,
            end,
            scene=None,
            loop=None,
            name=None,
            stiCond_save=None,
            wl_save=None,
            lum_save=None,
            eye_save=None,
            tF_save=None,
            dir_save=None,
            pha_save=None,
            brk_save=None,
        ):
            row = {
                "stiCond": stiCond_save,
                "AbsoluteStart_ms": start * 1000,
                "AbsoluteEnd_ms": end * 1000,
                "Duration_ms": (end - start) * 1000,
                "Wavelength": wl_save,
                "Temp_freq": tF_save,
                "Luminance": lum_save,
                "Eye": eye_save,
                "Direction": dir_save,
                "Phase": pha_save,
                "Break_stim": brk_save,
            }
            return row

        def startNewTimer():
            nonlocal timer_start, pic, in_timer_break, scene_defaultImage, current_bouncing_dir

            in_timer_break = False
            timer_start = now
            current_bouncing_dir = -1 + (current_timer % 2 != 0) * 2
            if current_scene.timer_update is not None:
                current_scene.layers = current_scene.timer_update(current_scene, current_timer)
                scene_defaultImage = (
                    self.superpose(current_scene.layers)
                    if len(current_scene.layers) > 1
                    else current_scene.layers[0]
                )
                pic = scene_defaultImage

        def startNewScene():
            nonlocal scene_start, in_scene_break, current_scene, current_timer, scene_defaultImage
            in_scene_break = False
            scene_start = now
            current_scene.timer = current_scene.timer or scene_timers
            current_scene.timer_iterations = current_scene.timer_iterations or timer_iterations
            current_scene.timer_break = current_scene.timer_break or timer_breaks

            if callable(current_scene.starting_lambda):
                current_scene.starting_lambda(current_scene)

            if (current_scene.update is None) and (current_scene.timer_update is None):
                if current_scene.layers is not None:
                    scene_defaultImage = (
                        self.superpose(current_scene.layers)
                        if len(current_scene.layers) > 1
                        else current_scene.layers[0]
                    )
            current_timer = 0
            startNewTimer()

        def endTimer():
            nonlocal current_timer, current_scene_idx, current_scene, current_loop, timer_over

            current_timer += 1
            if current_timer != current_scene.timer_iterations:
                startNewTimer()
            else:
                current_scene_idx += 1

                if current_scene_idx != scenes_per_loop:
                    current_scene = scenes[scene_order[current_scene_idx]]
                    current_scene.start_delay = current_scene.start_delay or start_delays
                    if current_scene.start_delay == 0.0:
                        startNewScene()
                    else:
                        startNewBreak()
                else:
                    current_loop += 1

                    if current_loop != iteration:
                        startNewLoop()
                    else:
                        timer_over = True
                        return True
            return False

        def startNewBreak():
            nonlocal in_scene_break, scene_break_start
            in_scene_break = True
            scene_break_start = now
            current_scene.starting_screen = (
                current_scene.starting_screen
                if current_scene.starting_screen is not None
                else starting_screens
            )
            current_scene.starting_screen_flat = (
                current_scene.starting_screen_flat or starting_screens_flat
            )

        def startNewLoop():
            nonlocal loop_start, current_scene_idx, scene_order, current_scene
            loop_start = now
            if random_order:
                np.random.shuffle(scene_order)
            current_scene_idx = 0
            current_scene = scenes[scene_order[current_scene_idx]]
            current_scene.start_delay = current_scene.start_delay or start_delays
            if current_scene.start_delay == 0.0:
                startNewScene()
            else:
                startNewBreak()

        startNewLoop()

        no_external_termination = (
            (lambda: not (self.is_ESC_pressed() | self.has_eye_recording_stopped()))
            if EyeTracker_linked
            else (lambda: not self.is_ESC_pressed())
        )

        frame_counter = 0
        run_start_time = time.time()
        while no_external_termination():
            frame_counter += 1
            pic = scene_defaultImage
            now = time.time()
            distortion_func = self.projected_image

            if in_scene_break:
                scene_break_elapsed_time = now - scene_break_start
                if scene_break_elapsed_time < current_scene.start_delay:
                    if current_scene.layers_brk is None:
                        pic = current_scene.starting_screen
                    else:
                        pic = current_scene.layers_brk[0] if isinstance(current_scene.layers_brk, list) else current_scene.layers_brk
                    if current_scene.starting_screen_flat:
                        distortion_func = self.flatten_image
                else:
                    startNewScene()

            if not in_scene_break:
                timer_elpased_time = now - timer_start

                if in_timer_break:
                    if timer_elpased_time < (current_scene.timer + current_scene.timer_break):
                        pic = scene_defaultImage
                    elif endTimer():
                        break
                else:
                    if timer_elpased_time < current_scene.timer:
                        if current_scene.update is not None:
                            progress = timer_elpased_time / current_scene.timer
                            position = current_bouncing_dir * (-0.5 + progress)
                            new_scene_layers = current_scene.update(
                                current_scene,
                                current_timer,
                                timer_elpased_time,
                                progress,
                                position,
                            )
                            pic = (
                                self.superpose(new_scene_layers)
                                if len(new_scene_layers) > 1
                                else new_scene_layers[0]
                            )
                            scene_defaultImage = pic
                    else:
                        in_timer_break = current_scene.timer_break != 0.0
                        logs.append(
                            createLogRow(
                                event="timer",
                                id=current_timer,
                                start=timer_start,
                                end=now,
                                scene=current_scene_idx,
                                loop=current_loop,
                                name=current_scene.name,
                                stiCond_save=current_scene.stiCond_save,
                                wl_save=current_scene.wl_save,
                                tF_save=current_scene.tF_save,
                                dir_save=current_scene.dir_save,
                                lum_save=current_scene.lum_save,
                                eye_save=current_scene.eye_save,
                                pha_save=current_scene.pha_save,
                                brk_save=current_scene.brk_save,
                            )
                        )
                        if in_timer_break:
                            if current_scene.brk_save == 1:
                                pic = scene_defaultImage
                            elif current_scene.layers_brk is not None:
                                pic = current_scene.layers_brk[0]
                                scene_defaultImage = pic
                            # else: keep last rendered frame as the break image
                        elif endTimer():
                            break

            self.display_image(pic, distortion_func)

        now = time.time()
        if not timer_over:
            logs.append(
                createLogRow(
                    event="timer",
                    id=current_timer,
                    start=timer_start,
                    end=now,
                    scene=current_scene_idx,
                    loop=current_loop,
                    name=current_bouncing_dir,
                    wl_save=current_scene.wl_save,
                    lum_save=current_scene.lum_save,
                    eye_save=current_scene.eye_save,
                    dir_save=current_scene.dir_save,
                    pha_save=current_scene.pha_save,
                    brk_save=current_scene.brk_save,
                )
            )

        print("mean fps " + str(frame_counter / (now - run_start_time)))
        return pd.DataFrame(logs)


@jit
def jit_flip_vertically(matrix):
    return matrix[::-1]


@jit
def jit_shift(pic, xshift=0, yshift=0):
    rolled = jnp.roll(pic, xshift, axis=0)
    rolled = jnp.roll(rolled, yshift, axis=1)
    return rolled


def jit_superpose(layers):
    result = layers[-1]
    for i in range(len(layers) - 2, -1, -1):
        layer = layers[i]
        mask = (layer != 0).any(axis=-1, keepdims=True) if layer.ndim == 3 else (layer != 0)
        result = jnp.where(mask, layer, result)
    return result


@jit
def jit_grey_to_rgb(pic):
    return jnp.repeat(pic[:, :, None], 3, axis=2)


@jit
def jit_image_to_bowl(img, rhos, phis, mask):
    img = img[phis, rhos, :]
    img = img * jnp.bitwise_and(mask[..., jnp.newaxis], 1)
    return img


@jit
def jit_rotate_sphere(img, rotation_coord):
    return img[rotation_coord[0], rotation_coord[1], :]


class BowlProjection:
    def __init__(
        self,
        stim_width,
        stim_height,
        stim_scale_x,
        stim_scale_y,
        fov_ele,
        is_360,
        rot_offset,
        margin,
        screen_res,
    ):

        self.stim_width = 360 if is_360 else stim_width
        self.stim_height = stim_height
        self.fov_ele = fov_ele

        self.stim_x = int(self.stim_width * stim_scale_x)
        self.stim_y = int(self.stim_height * stim_scale_y)

        projected_area = np.subtract(screen_res, np.array(margin) * 2)
        xmat, ymat = np.ogrid[:projected_area[0], :projected_area[1]]

        xproj_center = int(projected_area[0] / 2)
        yproj_center = int(projected_area[1] / 2 if is_360 else projected_area[1])

        yhalf_stim = int(self.stim_y / 2 if is_360 else self.stim_y)
        max_rhos_value = min(self.stim_x, self.stim_y) if is_360 else min(yhalf_stim, yhalf_stim)

        self.rhos = (
            np.around(
                (
                    np.sqrt((xmat - xproj_center) ** 2 + (ymat - yproj_center) ** 2)
                    * max_rhos_value
                    / min(xproj_center, yproj_center)
                )
            )
        ).astype(int)

        arct = np.arctan2(ymat - yproj_center, xmat - xproj_center)
        if not is_360:
            self.phis = (np.around(arct / np.pi * self.stim_x)).astype(int)
        else:
            self.phis = (
                np.around((arct + np.pi / 2) % (2 * np.pi) / (2 * np.pi) * self.stim_x)
            ).astype(int)

        self.mask = (
            (self.rhos <= max_rhos_value) & (self.rhos >= self.fov_ele * stim_scale_y)
        ).astype(np.uint8) * 255

        padding = ((margin[0], margin[0]), (margin[1], margin[1]))
        self.rhos = np.pad(self.rhos, padding, mode="constant", constant_values=0)
        self.phis = np.pad(self.phis, padding, mode="constant", constant_values=0)
        self.mask = np.pad(self.mask, padding, mode="constant", constant_values=0)

        self.flat_to_bowl = 0

        if rot_offset == (0, 0, 0):
            self.flat_to_bowl = lambda img: jit_image_to_bowl(img, self.rhos, self.phis, self.mask)
        else:
            roll, pitch, yaw = [jnp.deg2rad(a) for a in rot_offset]
            rx = jnp.array(
                [[1, 0, 0], [0, jnp.cos(roll), -jnp.sin(roll)], [0, jnp.sin(roll), jnp.cos(roll)]]
            )
            ry = jnp.array(
                [
                    [jnp.cos(pitch), 0, jnp.sin(pitch)],
                    [0, 1, 0],
                    [-jnp.sin(pitch), 0, jnp.cos(pitch)],
                ]
            )
            rz = jnp.array(
                [[jnp.cos(yaw), -jnp.sin(yaw), 0], [jnp.sin(yaw), jnp.cos(yaw), 0], [0, 0, 1]]
            )
            rmat = rz @ ry @ rx

            lon = jnp.linspace(-jnp.pi, jnp.pi, self.stim_x, endpoint=False)
            lat = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, self.stim_y, endpoint=False)
            lon, lat = jnp.meshgrid(lon, lat, indexing="xy")

            lon = lon.T
            lat = lat.T

            x = jnp.cos(lat) * jnp.cos(lon)
            y = jnp.cos(lat) * jnp.sin(lon)
            z = jnp.sin(lat)
            v = jnp.stack([x, y, z], -1)

            vr = v @ rmat.T

            lon_r = jnp.arctan2(vr[..., 1], vr[..., 0])
            lat_r = jnp.arcsin(vr[..., 2])

            ix = jnp.mod((lon_r + jnp.pi) / (2 * jnp.pi) * self.stim_x, self.stim_x - 1)
            iy = ((lat_r + jnp.pi / 2) / jnp.pi * self.stim_y).clip(0, self.stim_y - 1)

            self.rotation_coord = [jnp.round(ix).astype(int), jnp.round(iy).astype(int)]

            self.flat_to_bowl = lambda img: jit_image_to_bowl(
                jit_rotate_sphere(img, self.rotation_coord), self.rhos, self.phis, self.mask
            )
