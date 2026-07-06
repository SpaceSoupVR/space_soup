use fontdue::Font as FontdueFont;
use std::ops::Deref;
use std::sync::Arc;

use crate::ui2d::Color;

#[derive(Debug)]
pub struct Font(pub(crate) FontdueFont);

impl Font {
    pub fn new(bytes: &[u8]) -> Self {
        let settings = fontdue::FontSettings::default();
        let font = FontdueFont::from_bytes(bytes, settings).expect("failed to parse font bytes");
        Self(font)
    }

    pub fn rasterize(&self, c: char, px: f32) -> (fontdue::Metrics, Vec<u8>) {
        self.0.rasterize(c, px)
    }
}

impl Deref for Font {
    type Target = FontdueFont;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Align {
    #[default]
    Left,
    Center,
    Right,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Span {
    pub text: String,
    pub font: Arc<Font>,
    pub font_size: f32,
    pub color: Color,
    pub align: Align,
}

impl PartialEq for Font {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self, other)
    }
}

impl Span {
    pub fn new(text: impl Into<String>, font: Arc<Font>, font_size: f32, color: Color) -> Self {
        Self {
            text: text.into(),
            font,
            font_size,
            color,
            align: Align::default(),
        }
    }

    pub fn with_align(mut self, align: Align) -> Self {
        self.align = align;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Character(
    pub char,
    pub (f32, f32, f32, f32),
    pub Arc<Font>,
    pub Color,
    pub f32,
    pub f32,
    pub f32,
);

#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    lines: Vec<(f32, (f32, f32), Vec<Character>)>,
    width: f32,
}

impl PartialEq for Character {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
            && self.1 == other.1
            && Arc::ptr_eq(&self.2, &other.2)
            && self.3 == other.3
            && self.4 == other.4
            && self.5 == other.5
            && self.6 == other.6
    }
}

impl Text {
    pub fn new(spans: Vec<Span>, max_width: f32) -> Self {
        let mut lines: Vec<(f32, (f32, f32), Vec<Character>)> = Vec::new();

        let mut cursor_x = 0.0_f32;
        let mut cursor_y = 0.0_f32;
        let mut line_chars: Vec<Character> = Vec::new();
        let mut line_height = 0.0_f32;
        let mut content_width = 0.0_f32;

        for span in &spans {
            line_height = line_height.max(span.font_size * 1.2);
            let descent = span
                .font
                .horizontal_line_metrics(span.font_size)
                .map(|lm| lm.descent)
                .unwrap_or(0.0);

            for c in span.text.chars() {
                if c == '\n' {
                    lines.push((
                        cursor_y,
                        (cursor_x, line_height),
                        std::mem::take(&mut line_chars),
                    ));
                    content_width = content_width.max(cursor_x);
                    cursor_x = 0.0;
                    cursor_y += line_height;
                    line_height = span.font_size * 1.2;
                    continue;
                }

                let metrics = span.font.metrics(c, span.font_size);
                let advance = metrics.advance_width;
                let bounds = metrics.bounds;

                if cursor_x + advance > max_width && cursor_x > 0.0 {
                    lines.push((
                        cursor_y,
                        (cursor_x, line_height),
                        std::mem::take(&mut line_chars),
                    ));
                    content_width = content_width.max(cursor_x);
                    cursor_x = 0.0;
                    cursor_y += line_height;
                }

                let x = cursor_x + bounds.xmin;
                let y = cursor_y + line_height + descent - bounds.ymin - bounds.height;

                line_chars.push(Character(
                    c,
                    (x, y, bounds.width, bounds.height),
                    span.font.clone(),
                    span.color,
                    0.0,
                    0.0,
                    span.font_size,
                ));

                cursor_x += advance;
            }
        }

        if !line_chars.is_empty() || lines.is_empty() {
            content_width = content_width.max(cursor_x);
            lines.push((cursor_y, (cursor_x, line_height.max(1.0)), line_chars));
        }

        Self {
            lines,
            width: content_width,
        }
    }

    pub fn lines(&self) -> &[(f32, (f32, f32), Vec<Character>)] {
        &self.lines
    }

    pub fn width(&self) -> f32 {
        self.width
    }

    pub fn height(&self) -> f32 {
        self.lines
            .iter()
            .map(|(y, (_, h), _)| y + h)
            .fold(0.0, f32::max)
    }
}
