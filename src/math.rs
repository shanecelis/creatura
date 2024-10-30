#[cfg(feature = "avian")]
mod avian_types {
    pub use avian3d::math::Quaternion;
    pub use avian3d::math::Scalar;
    pub use avian3d::math::Vector3;
}
#[cfg(feature = "avian")]
pub use avian_types::*;

#[cfg(not(feature = "avian"))]
mod default_types {
    use bevy::prelude::*;
    pub type Quaternion = Quat;
    pub type Vector3 = Vec3;
    pub type Scalar = f32;
}
#[cfg(not(feature = "avian"))]
pub use default_types::*;
