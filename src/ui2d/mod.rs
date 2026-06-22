use std::sync::Arc;

pub mod shape;
pub mod text;
pub mod renderer;
pub mod overlay;

pub use shape::Shape as ShapeType;
pub use text::{Font, Text, Span, Align, Character};
pub use renderer::{Renderer as Ui2dRenderer, Atlas};
pub use overlay::Overlay;

pub use image::RgbaImage;

#[derive(Default, Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct Color(pub u8, pub u8, pub u8, pub u8);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shape {
    pub shape: ShapeType,
    pub color: Color,
}

#[derive(Clone, PartialEq)]
pub struct Image {
    pub shape: ShapeType,
    pub image: Arc<RgbaImage>,
    pub color: Option<Color>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Image")
            .field("shape", &self.shape)
            .field("image", &"Arc<RgbaImage>")
            .field("color", &self.color)
            .finish()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Item {
    Shape(Shape),
    Image(Image),
    Text(Text),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Area {
    pub offset: (f32, f32),
    pub bounds: Option<(f32, f32, f32, f32)>,
}
