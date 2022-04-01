use core::{
    any::TypeId,
    marker::PhantomData,
    ops::{Add, AddAssign, Sub, SubAssign},
};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};
#[cfg(feature = "random")]
use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform, Uniform, UniformSampler},
        Distribution, Standard,
    },
    Rng,
};

#[cfg(feature = "random")]
use crate::num::{Cbrt, Powi, Sqrt};

use crate::{
    angle::{FromAngle, RealAngle, SignedAngle},
    clamp, clamp_assign, contrast_ratio,
    convert::FromColorUnclamped,
    encoding::Srgb,
    num::{Arithmetics, IsValidDivisor, MinMax, One, Real, Zero},
    rgb::{Rgb, RgbSpace, RgbStandard},
    stimulus::{FromStimulus, Stimulus},
    Alpha, Clamp, ClampAssign, FromColor, GetHue, Hsv, IsWithinBounds, Lighten, LightenAssign, Mix,
    MixAssign, RelativeContrast, RgbHue, Saturate, SaturateAssign, SetHue, ShiftHue,
    ShiftHueAssign, WithHue, Xyz,
};

/// Linear HSL with an alpha component. See the [`Hsla` implementation in
/// `Alpha`](crate::Alpha#Hsla).
pub type Hsla<S = Srgb, T = f32> = Alpha<Hsl<S, T>, T>;

/// HSL color space.
///
/// The HSL color space can be seen as a cylindrical version of
/// [RGB](crate::rgb::Rgb), where the `hue` is the angle around the color
/// cylinder, the `saturation` is the distance from the center, and the
/// `lightness` is the height from the bottom. Its composition makes it
/// especially good for operations like changing green to red, making a color
/// more gray, or making it darker.
///
/// HSL component values are typically real numbers (such as floats), but may
/// also be converted to and from `u8` for storage and interoperability
/// purposes. The hue is then within the range `[0, 255]`.
///
/// ```
/// use approx::assert_relative_eq;
/// use palette::Hsl;
///
/// let hsl_u8 = Hsl::new_srgb(128u8, 85, 51);
/// let hsl_f32 = hsl_u8.into_format::<f32>();
///
/// assert_relative_eq!(hsl_f32, Hsl::new(180.0, 1.0 / 3.0, 0.2));
/// ```
///
/// See [HSV](crate::Hsv) for a very similar color space, with brightness
/// instead of lightness.
#[derive(ArrayCast, FromColorUnclamped, WithAlpha)]
#[cfg_attr(feature = "serializing", derive(Serialize, Deserialize))]
#[palette(
    palette_internal,
    rgb_standard = "S",
    component = "T",
    skip_derives(Rgb, Hsv, Hsl)
)]
#[repr(C)]
pub struct Hsl<S = Srgb, T = f32> {
    /// The hue of the color, in degrees. Decides if it's red, blue, purple,
    /// etc.
    #[palette(unsafe_same_layout_as = "T")]
    pub hue: RgbHue<T>,

    /// The colorfulness of the color. 0.0 gives gray scale colors and 1.0 will
    /// give absolutely clear colors.
    pub saturation: T,

    /// Decides how light the color will look. 0.0 will be black, 0.5 will give
    /// a clear color, and 1.0 will give white.
    pub lightness: T,

    /// The white point and RGB primaries this color is adapted to. The default
    /// is the sRGB standard.
    #[cfg_attr(feature = "serializing", serde(skip))]
    #[palette(unsafe_zero_sized)]
    pub standard: PhantomData<S>,
}

impl<S, T: Copy> Copy for Hsl<S, T> {}

impl<S, T: Clone> Clone for Hsl<S, T> {
    fn clone(&self) -> Hsl<S, T> {
        Hsl {
            hue: self.hue.clone(),
            saturation: self.saturation.clone(),
            lightness: self.lightness.clone(),
            standard: PhantomData,
        }
    }
}

impl<T> Hsl<Srgb, T> {
    /// Create an sRGB HSL color. This method can be used instead of `Hsl::new`
    /// to help type inference.
    pub fn new_srgb<H: Into<RgbHue<T>>>(hue: H, saturation: T, lightness: T) -> Self {
        Self::new_const(hue.into(), saturation, lightness)
    }

    /// Create an sRGB HSL color. This is the same as `Hsl::new_srgb` without
    /// the generic hue type. It's temporary until `const fn` supports traits.
    pub const fn new_srgb_const(hue: RgbHue<T>, saturation: T, lightness: T) -> Self {
        Self::new_const(hue, saturation, lightness)
    }
}

impl<S, T> Hsl<S, T> {
    /// Create an HSL color.
    pub fn new<H: Into<RgbHue<T>>>(hue: H, saturation: T, lightness: T) -> Self {
        Self::new_const(hue.into(), saturation, lightness)
    }

    /// Create an HSL color. This is the same as `Hsl::new` without the generic
    /// hue type. It's temporary until `const fn` supports traits.
    pub const fn new_const(hue: RgbHue<T>, saturation: T, lightness: T) -> Self {
        Hsl {
            hue,
            saturation,
            lightness,
            standard: PhantomData,
        }
    }

    /// Convert into another component type.
    pub fn into_format<U>(self) -> Hsl<S, U>
    where
        U: FromStimulus<T> + FromAngle<T>,
    {
        Hsl {
            hue: self.hue.into_format(),
            saturation: U::from_stimulus(self.saturation),
            lightness: U::from_stimulus(self.lightness),
            standard: PhantomData,
        }
    }

    /// Convert from another component type.
    pub fn from_format<U>(color: Hsl<S, U>) -> Self
    where
        T: FromStimulus<U> + FromAngle<U>,
    {
        color.into_format()
    }

    /// Convert to a `(hue, saturation, lightness)` tuple.
    pub fn into_components(self) -> (RgbHue<T>, T, T) {
        (self.hue, self.saturation, self.lightness)
    }

    /// Convert from a `(hue, saturation, lightness)` tuple.
    pub fn from_components<H: Into<RgbHue<T>>>((hue, saturation, lightness): (H, T, T)) -> Self {
        Self::new(hue, saturation, lightness)
    }

    #[inline]
    fn reinterpret_as<St: RgbStandard<T>>(self) -> Hsl<St, T> {
        Hsl {
            hue: self.hue,
            saturation: self.saturation,
            lightness: self.lightness,
            standard: PhantomData,
        }
    }
}

impl<S, T> Hsl<S, T>
where
    T: Stimulus,
{
    /// Return the `saturation` value minimum.
    pub fn min_saturation() -> T {
        T::zero()
    }

    /// Return the `saturation` value maximum.
    pub fn max_saturation() -> T {
        T::max_intensity()
    }

    /// Return the `lightness` value minimum.
    pub fn min_lightness() -> T {
        T::zero()
    }

    /// Return the `lightness` value maximum.
    pub fn max_lightness() -> T {
        T::max_intensity()
    }
}

///<span id="Hsla"></span>[`Hsla`](crate::Hsla) implementations.
impl<T, A> Alpha<Hsl<Srgb, T>, A> {
    /// Create an sRGB HSL color with transparency. This method can be used
    /// instead of `Hsla::new` to help type inference.
    pub fn new_srgb<H: Into<RgbHue<T>>>(hue: H, saturation: T, lightness: T, alpha: A) -> Self {
        Self::new_const(hue.into(), saturation, lightness, alpha)
    }

    /// Create an sRGB HSL color with transparency. This is the same as
    /// `Hsla::new_srgb` without the generic hue type. It's temporary until
    /// `const fn` supports traits.
    pub const fn new_srgb_const(hue: RgbHue<T>, saturation: T, lightness: T, alpha: A) -> Self {
        Self::new_const(hue, saturation, lightness, alpha)
    }
}

///<span id="Hsla"></span>[`Hsla`](crate::Hsla) implementations.
impl<S, T, A> Alpha<Hsl<S, T>, A> {
    /// Create an HSL color with transparency.
    pub fn new<H: Into<RgbHue<T>>>(hue: H, saturation: T, lightness: T, alpha: A) -> Self {
        Self::new_const(hue.into(), saturation, lightness, alpha)
    }

    /// Create an HSL color with transparency. This is the same as `Hsla::new`
    /// without the generic hue type. It's temporary until `const fn` supports
    /// traits.
    pub const fn new_const(hue: RgbHue<T>, saturation: T, lightness: T, alpha: A) -> Self {
        Alpha {
            color: Hsl::new_const(hue, saturation, lightness),
            alpha,
        }
    }
    /// Convert into another component type.
    pub fn into_format<U, B>(self) -> Alpha<Hsl<S, U>, B>
    where
        U: FromStimulus<T> + FromAngle<T>,
        B: FromStimulus<A>,
    {
        Alpha {
            color: self.color.into_format(),
            alpha: B::from_stimulus(self.alpha),
        }
    }

    /// Convert from another component type.
    pub fn from_format<U, B>(color: Alpha<Hsl<S, U>, B>) -> Self
    where
        T: FromStimulus<U> + FromAngle<U>,
        A: FromStimulus<B>,
    {
        color.into_format()
    }

    /// Convert to a `(hue, saturation, lightness, alpha)` tuple.
    pub fn into_components(self) -> (RgbHue<T>, T, T, A) {
        (
            self.color.hue,
            self.color.saturation,
            self.color.lightness,
            self.alpha,
        )
    }

    /// Convert from a `(hue, saturation, lightness, alpha)` tuple.
    pub fn from_components<H: Into<RgbHue<T>>>(
        (hue, saturation, lightness, alpha): (H, T, T, A),
    ) -> Self {
        Self::new(hue, saturation, lightness, alpha)
    }
}

impl<S1, S2, T> FromColorUnclamped<Hsl<S1, T>> for Hsl<S2, T>
where
    S1: RgbStandard<T>,
    S2: RgbStandard<T>,
    S1::Space: RgbSpace<T, WhitePoint = <S2::Space as RgbSpace<T>>::WhitePoint>,
    Rgb<S1, T>: FromColorUnclamped<Hsl<S1, T>>,
    Rgb<S2, T>: FromColorUnclamped<Rgb<S1, T>>,
    Self: FromColorUnclamped<Rgb<S2, T>>,
{
    fn from_color_unclamped(hsl: Hsl<S1, T>) -> Self {
        if TypeId::of::<S1>() == TypeId::of::<S2>() {
            hsl.reinterpret_as()
        } else {
            let rgb = Rgb::<S1, T>::from_color_unclamped(hsl);
            let converted_rgb = Rgb::<S2, T>::from_color_unclamped(rgb);
            Self::from_color_unclamped(converted_rgb)
        }
    }
}

impl<S, T> FromColorUnclamped<Rgb<S, T>> for Hsl<S, T>
where
    T: Real + Zero + One + MinMax + Arithmetics + PartialOrd + Clone,
{
    fn from_color_unclamped(mut rgb: Rgb<S, T>) -> Self {
        // Avoid negative numbers
        rgb.red = rgb.red.max(T::zero());
        rgb.green = rgb.green.max(T::zero());
        rgb.blue = rgb.blue.max(T::zero());

        let (max, min, sep, coeff) = {
            let (max, min, sep, coeff) = if rgb.red > rgb.green {
                (
                    rgb.red.clone(),
                    rgb.green.clone(),
                    rgb.green.clone() - &rgb.blue,
                    T::zero(),
                )
            } else {
                (
                    rgb.green.clone(),
                    rgb.red.clone(),
                    rgb.blue.clone() - &rgb.red,
                    T::from_f64(2.0),
                )
            };
            if rgb.blue > max {
                (rgb.blue, min, rgb.red - rgb.green, T::from_f64(4.0))
            } else {
                let min_val = if rgb.blue < min { rgb.blue } else { min };
                (max, min_val, sep, coeff)
            }
        };

        let mut h = T::zero();
        let mut s = T::zero();

        let sum = max.clone() + &min;
        let l = sum.clone() / T::from_f64(2.0);
        if max != min {
            let d = max - min;
            s = if sum > T::one() {
                d.clone() / (T::from_f64(2.0) - sum)
            } else {
                d.clone() / sum
            };
            h = ((sep / d) + coeff) * T::from_f64(60.0);
        };

        Hsl {
            hue: h.into(),
            saturation: s,
            lightness: l,
            standard: PhantomData,
        }
    }
}

impl<S, T> FromColorUnclamped<Hsv<S, T>> for Hsl<S, T>
where
    T: Real + Zero + One + IsValidDivisor + Arithmetics + PartialOrd,
{
    fn from_color_unclamped(hsv: Hsv<S, T>) -> Self {
        let x = (T::from_f64(2.0) - &hsv.saturation) * &hsv.value;
        let saturation = if !hsv.value.is_valid_divisor() {
            T::zero()
        } else if x < T::one() {
            if x.is_valid_divisor() {
                hsv.saturation * hsv.value / &x
            } else {
                T::zero()
            }
        } else {
            let denom = T::from_f64(2.0) - &x;
            if denom.is_valid_divisor() {
                hsv.saturation * hsv.value / denom
            } else {
                T::zero()
            }
        };

        Hsl {
            hue: hsv.hue,
            saturation,
            lightness: x / T::from_f64(2.0),
            standard: PhantomData,
        }
    }
}

impl<S, T, H: Into<RgbHue<T>>> From<(H, T, T)> for Hsl<S, T> {
    fn from(components: (H, T, T)) -> Self {
        Self::from_components(components)
    }
}

impl<S, T> From<Hsl<S, T>> for (RgbHue<T>, T, T) {
    fn from(color: Hsl<S, T>) -> (RgbHue<T>, T, T) {
        color.into_components()
    }
}

impl<S, T, H: Into<RgbHue<T>>, A> From<(H, T, T, A)> for Alpha<Hsl<S, T>, A> {
    fn from(components: (H, T, T, A)) -> Self {
        Self::from_components(components)
    }
}

impl<S, T, A> From<Alpha<Hsl<S, T>, A>> for (RgbHue<T>, T, T, A) {
    fn from(color: Alpha<Hsl<S, T>, A>) -> (RgbHue<T>, T, T, A) {
        color.into_components()
    }
}

impl<S, T> IsWithinBounds for Hsl<S, T>
where
    T: Stimulus + PartialOrd,
{
    #[rustfmt::skip]
    #[inline]
    fn is_within_bounds(&self) -> bool {
        self.saturation >= Self::min_saturation() && self.saturation <= Self::max_saturation() &&
        self.lightness >= Self::min_lightness() && self.lightness <= Self::max_lightness()
    }
}

impl<S, T> Clamp for Hsl<S, T>
where
    T: Stimulus + PartialOrd,
{
    #[inline]
    fn clamp(self) -> Self {
        Self::new(
            self.hue,
            clamp(
                self.saturation,
                Self::min_saturation(),
                Self::max_saturation(),
            ),
            clamp(self.lightness, Self::min_lightness(), Self::max_lightness()),
        )
    }
}

impl<S, T> ClampAssign for Hsl<S, T>
where
    T: Stimulus + PartialOrd,
{
    #[inline]
    fn clamp_assign(&mut self) {
        clamp_assign(
            &mut self.saturation,
            Self::min_saturation(),
            Self::max_saturation(),
        );
        clamp_assign(
            &mut self.lightness,
            Self::min_lightness(),
            Self::max_lightness(),
        );
    }
}

impl_mix_hue!(Hsl<S> {saturation, lightness} phantom: standard);
impl_lighten!(Hsl<S> increase {lightness => [Self::min_lightness(), Self::max_lightness()]} other {hue, saturation} phantom: standard where T: Stimulus);
impl_saturate!(Hsl<S> increase {saturation => [Self::min_saturation(), Self::max_saturation()]} other {hue, lightness} phantom: standard where T: Stimulus);

impl<S, T> GetHue for Hsl<S, T>
where
    T: Zero + PartialOrd + Clone,
{
    type Hue = RgbHue<T>;

    #[inline]
    fn get_hue(&self) -> Option<RgbHue<T>> {
        if self.saturation <= T::zero() {
            None
        } else {
            Some(self.hue.clone())
        }
    }
}

impl<S, T, H> WithHue<H> for Hsl<S, T>
where
    H: Into<RgbHue<T>>,
{
    #[inline]
    fn with_hue(mut self, hue: H) -> Self {
        self.hue = hue.into();
        self
    }
}

impl<S, T, H> SetHue<H> for Hsl<S, T>
where
    H: Into<RgbHue<T>>,
{
    #[inline]
    fn set_hue(&mut self, hue: H) {
        self.hue = hue.into();
    }
}

impl<S, T> ShiftHue for Hsl<S, T>
where
    T: Add<Output = T>,
{
    type Scalar = T;

    #[inline]
    fn shift_hue(mut self, amount: Self::Scalar) -> Self {
        self.hue = self.hue + amount;
        self
    }
}

impl<S, T> ShiftHueAssign for Hsl<S, T>
where
    T: AddAssign,
{
    type Scalar = T;

    #[inline]
    fn shift_hue_assign(&mut self, amount: Self::Scalar) {
        self.hue += amount;
    }
}

impl<S, T> Default for Hsl<S, T>
where
    T: Stimulus,
    RgbHue<T>: Default,
{
    fn default() -> Hsl<S, T> {
        Hsl::new(
            RgbHue::default(),
            Self::min_saturation(),
            Self::min_lightness(),
        )
    }
}

impl_color_add!(Hsl<S, T>, [hue, saturation, lightness], standard);
impl_color_sub!(Hsl<S, T>, [hue, saturation, lightness], standard);

impl_array_casts!(Hsl<S, T>, [T; 3]);

impl_eq_hue!(Hsl<S>, RgbHue, [hue, saturation, lightness]);

impl<S, T> RelativeContrast for Hsl<S, T>
where
    T: Real + PartialOrd + Arithmetics,
    S: RgbStandard<T>,
    Xyz<<S::Space as RgbSpace<T>>::WhitePoint, T>: FromColor<Self>,
{
    type Scalar = T;

    #[inline]
    fn get_contrast_ratio(self, other: Self) -> T {
        let xyz1 = Xyz::from_color(self);
        let xyz2 = Xyz::from_color(other);

        contrast_ratio(xyz1.y, xyz2.y)
    }
}

#[cfg(feature = "random")]
impl<S, T> Distribution<Hsl<S, T>> for Standard
where
    T: Real + Cbrt + Sqrt + Arithmetics + PartialOrd,
    Standard: Distribution<T> + Distribution<RgbHue<T>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Hsl<S, T> {
        crate::random_sampling::sample_hsl(rng.gen::<RgbHue<T>>(), rng.gen(), rng.gen())
    }
}

#[cfg(feature = "random")]
pub struct UniformHsl<S, T>
where
    T: SampleUniform,
{
    hue: crate::hues::UniformRgbHue<T>,
    u1: Uniform<T>,
    u2: Uniform<T>,
    space: PhantomData<S>,
}

#[cfg(feature = "random")]
impl<S, T> SampleUniform for Hsl<S, T>
where
    T: Real + Cbrt + Sqrt + Powi + Arithmetics + PartialOrd + Clone + SampleUniform,
    RgbHue<T>: SampleBorrow<RgbHue<T>>,
    crate::hues::UniformRgbHue<T>: UniformSampler<X = RgbHue<T>>,
{
    type Sampler = UniformHsl<S, T>;
}

#[cfg(feature = "random")]
impl<S, T> UniformSampler for UniformHsl<S, T>
where
    T: Real + Cbrt + Sqrt + Powi + Arithmetics + PartialOrd + Clone + SampleUniform,
    RgbHue<T>: SampleBorrow<RgbHue<T>>,
    crate::hues::UniformRgbHue<T>: UniformSampler<X = RgbHue<T>>,
{
    type X = Hsl<S, T>;

    fn new<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        use crate::random_sampling::invert_hsl_sample;

        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        let (r1_min, r2_min) = invert_hsl_sample(low.saturation, low.lightness);
        let (r1_max, r2_max) = invert_hsl_sample(high.saturation, high.lightness);

        UniformHsl {
            hue: crate::hues::UniformRgbHue::new(low.hue, high.hue),
            u1: Uniform::new::<_, T>(r1_min, r1_max),
            u2: Uniform::new::<_, T>(r2_min, r2_max),
            space: PhantomData,
        }
    }

    fn new_inclusive<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        use crate::random_sampling::invert_hsl_sample;

        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        let (r1_min, r2_min) = invert_hsl_sample(low.saturation, low.lightness);
        let (r1_max, r2_max) = invert_hsl_sample(high.saturation, high.lightness);

        UniformHsl {
            hue: crate::hues::UniformRgbHue::new_inclusive(low.hue, high.hue),
            u1: Uniform::new_inclusive::<_, T>(r1_min, r1_max),
            u2: Uniform::new_inclusive::<_, T>(r2_min, r2_max),
            space: PhantomData,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Hsl<S, T> {
        crate::random_sampling::sample_hsl(
            self.hue.sample(rng),
            self.u1.sample(rng),
            self.u2.sample(rng),
        )
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<S, T> bytemuck::Zeroable for Hsl<S, T> where T: bytemuck::Zeroable {}

#[cfg(feature = "bytemuck")]
unsafe impl<S: 'static, T> bytemuck::Pod for Hsl<S, T> where T: bytemuck::Pod {}

#[cfg(test)]
mod test {
    use super::Hsl;
    use crate::{FromColor, Hsv, Srgb};

    #[test]
    fn red() {
        let a = Hsl::from_color(Srgb::new(1.0, 0.0, 0.0));
        let b = Hsl::new_srgb(0.0, 1.0, 0.5);
        let c = Hsl::from_color(Hsv::new_srgb(0.0, 1.0, 1.0));

        assert_relative_eq!(a, b);
        assert_relative_eq!(a, c);
    }

    #[test]
    fn orange() {
        let a = Hsl::from_color(Srgb::new(1.0, 0.5, 0.0));
        let b = Hsl::new_srgb(30.0, 1.0, 0.5);
        let c = Hsl::from_color(Hsv::new_srgb(30.0, 1.0, 1.0));

        assert_relative_eq!(a, b);
        assert_relative_eq!(a, c);
    }

    #[test]
    fn green() {
        let a = Hsl::from_color(Srgb::new(0.0, 1.0, 0.0));
        let b = Hsl::new_srgb(120.0, 1.0, 0.5);
        let c = Hsl::from_color(Hsv::new_srgb(120.0, 1.0, 1.0));

        assert_relative_eq!(a, b);
        assert_relative_eq!(a, c);
    }

    #[test]
    fn blue() {
        let a = Hsl::from_color(Srgb::new(0.0, 0.0, 1.0));
        let b = Hsl::new_srgb(240.0, 1.0, 0.5);
        let c = Hsl::from_color(Hsv::new_srgb(240.0, 1.0, 1.0));

        assert_relative_eq!(a, b);
        assert_relative_eq!(a, c);
    }

    #[test]
    fn purple() {
        let a = Hsl::from_color(Srgb::new(0.5, 0.0, 1.0));
        let b = Hsl::new_srgb(270.0, 1.0, 0.5);
        let c = Hsl::from_color(Hsv::new_srgb(270.0, 1.0, 1.0));

        assert_relative_eq!(a, b);
        assert_relative_eq!(a, c);
    }

    #[test]
    fn ranges() {
        assert_ranges! {
            Hsl<crate::encoding::Srgb, f64>;
            clamped {
                saturation: 0.0 => 1.0,
                lightness: 0.0 => 1.0
            }
            clamped_min {}
            unclamped {
                hue: -360.0 => 360.0
            }
        }
    }

    raw_pixel_conversion_tests!(Hsl<crate::encoding::Srgb>: hue, saturation, lightness);
    raw_pixel_conversion_fail_tests!(Hsl<crate::encoding::Srgb>: hue, saturation, lightness);

    #[test]
    fn check_min_max_components() {
        use crate::encoding::Srgb;

        assert_relative_eq!(Hsl::<Srgb>::min_saturation(), 0.0);
        assert_relative_eq!(Hsl::<Srgb>::min_lightness(), 0.0);
        assert_relative_eq!(Hsl::<Srgb>::max_saturation(), 1.0);
        assert_relative_eq!(Hsl::<Srgb>::max_lightness(), 1.0);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn serialize() {
        let serialized = ::serde_json::to_string(&Hsl::new_srgb(0.3, 0.8, 0.1)).unwrap();

        assert_eq!(
            serialized,
            r#"{"hue":0.3,"saturation":0.8,"lightness":0.1}"#
        );
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn deserialize() {
        let deserialized: Hsl =
            ::serde_json::from_str(r#"{"hue":0.3,"saturation":0.8,"lightness":0.1}"#).unwrap();

        assert_eq!(deserialized, Hsl::new(0.3, 0.8, 0.1));
    }

    #[cfg(feature = "random")]
    test_uniform_distribution! {
        Hsl<crate::encoding::Srgb, f32> as crate::rgb::Rgb {
            red: (0.0, 1.0),
            green: (0.0, 1.0),
            blue: (0.0, 1.0)
        },
        min: Hsl::new(0.0f32, 0.0, 0.0),
        max: Hsl::new(360.0, 1.0, 1.0)
    }

    /// Sanity check to make sure the test doesn't start accepting known
    /// non-uniform distributions.
    #[cfg(feature = "random")]
    #[test]
    #[should_panic(expected = "is not uniform enough")]
    fn uniform_distribution_fail() {
        use rand::Rng;

        const BINS: usize = crate::random_sampling::test_utils::BINS;
        const SAMPLES: usize = crate::random_sampling::test_utils::SAMPLES;

        let mut red = [0; BINS];
        let mut green = [0; BINS];
        let mut blue = [0; BINS];

        let mut rng = rand_mt::Mt::new(1234); // We want the same seed on every run to avoid random fails

        for _ in 0..SAMPLES {
            let color = Hsl::<crate::encoding::Srgb, f32>::new(
                rng.gen::<f32>() * 360.0,
                rng.gen(),
                rng.gen(),
            );
            let color: crate::rgb::Rgb = crate::IntoColor::into_color(color);
            red[((color.red * BINS as f32) as usize).min(9)] += 1;
            green[((color.green * BINS as f32) as usize).min(9)] += 1;
            blue[((color.blue * BINS as f32) as usize).min(9)] += 1;
        }

        assert_uniform_distribution!(red);
        assert_uniform_distribution!(green);
        assert_uniform_distribution!(blue);
    }
}
