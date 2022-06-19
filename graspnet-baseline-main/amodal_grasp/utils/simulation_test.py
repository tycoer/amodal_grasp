from utils.simulation import ClutterRemovalSim
from utils.perception import *
from utils.transform import Rotation, Transform






class ClutterRemovalSimWithCategory(ClutterRemovalSim):
    def __init__(self,
                 scene,
                 object_set,
                 gui=True,
                 seed=None,
                 add_noise=False,
                 sideview=False,
                 save_dir=None,
                 save_freq=8,
                 category='bowl'):
        self.category = category
        super().__init__(scene=scene,
                         object_set=object_set,
                         gui=gui,
                         add_noise=add_noise,
                         seed=seed,
                         sideview=sideview,
                         save_dir=save_dir,
                         save_freq=save_freq,
                         )

    def discover_objects(self):
        root = self.urdf_root / self.object_set
        # self.object_urdfs = [f for f in root.iterdir() if f.suffix == ".urdf"]
        self.object_urdfs_without_category = [f for f in root.iterdir() if f.suffix == ".urdf" and self.category not in str(f)] # tycoer
        # tycoer
        self.object_urdfs_with_category = [f for f in root.iterdir() if f.suffix == ".urdf" and self.category in str(f)]


    def generate_pile_scene(self, object_count, table_height):
        # place box
        urdf = self.urdf_root / "setup" / "box.urdf"
        pose = Transform(Rotation.identity(), np.r_[0.02, 0.02, table_height])
        box = self.world.load_urdf(urdf, pose, scale=1.3)

        # drop objects\
        # tycoer
        if object_count < 2:
            object_count = 2
        urdf_with_category_num = np.random.randint(1, object_count)
        urdf_without_category_num = object_count - urdf_with_category_num

        urdfs = self.rng.choice(self.object_urdfs_with_category, size=urdf_with_category_num).tolist() + \
                self.rng.choice(self.object_urdfs_without_category, size=urdf_without_category_num).tolist()

        ###############
        # urdfs = self.rng.choice(self.object_urdfs, size=object_count)
        for urdf in urdfs:
            rotation = Rotation.random(random_state=self.rng)
            xy = self.rng.uniform(1.0 / 3.0 * self.size, 2.0 / 3.0 * self.size, 2)
            pose = Transform(rotation, np.r_[xy, table_height + 0.2])
            scale = self.rng.uniform(0.8, 1.0)
            self.world.load_urdf(urdf, pose, scale=self.global_scaling * scale)
            self.wait_for_objects_to_rest(timeout=1.0)
        # remove box
        self.world.remove_body(box)
        self.remove_and_wait()