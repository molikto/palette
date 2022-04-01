use core::{
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

use crate::{
    angle::{RealAngle, SignedAngle},
    clamp, clamp_assign, clamp_min, clamp_min_assign,
    color_difference::{get_ciede_difference, ColorDifference, LabColorDiff},
    contrast_ratio,
    convert::FromColorUnclamped,
    num::{Abs, Arithmetics, Exp, Hypot, MinMax, One, Powi, Real, Sqrt, Trigonometry, Zero},
    white_point::D65,
    Alpha, Clamp, ClampAssign, FromColor, GetHue, IsWithinBounds, Lab, LabHue, Lighten,
    LightenAssign, Mix, MixAssign, RelativeContrast, Saturate, SaturateAssign, SetHue, ShiftHue,
    ShiftHueAssign, WithHue, Xyz,
};

/// CIE L\*C\*h° with an alpha component. See the [`Lcha` implementation in
/// `Alpha`](crate::Alpha#Lcha).
pub type Lcha<Wp = D65, T = f32> = Alpha<Lch<Wp, T>, T>;

/// CIE L\*C\*h°, a polar version of [CIE L\*a\*b\*](crate::Lab).
///
/// L\*C\*h° shares its range and perceptual uniformity with L\*a\*b\*, but
/// it's a cylindrical color space, like [HSL](crate::Hsl) and
/// [HSV](crate::Hsv). This gives it the same ability to directly change
/// the hue and colorfulness of a color, while preserving other visual aspects.
#[derive(ArrayCast, FromColorUnclamped, WithAlpha)]
#[cfg_attr(feature = "serializing", derive(Serialize, Deserialize))]
#[palette(
    palette_internal,
    white_point = "Wp",
    component = "T",
    skip_derives(Lab, Lch)
)]
#[repr(C)]
pub struct Lch<Wp = D65, T = f32> {
    /// L\* is the lightness of the color. 0.0 gives absolute black and 100.0
    /// gives the brightest white.
    pub l: T,

    /// C\* is the colorfulness of the color. It's similar to saturation. 0.0
    /// gives gray scale colors, and numbers around 128-181 gives fully
    /// saturated colors. The upper limit of 128 should
    /// include the whole L\*a\*b\* space and some more.
    pub chroma: T,

    /// The hue of the color, in degrees. Decides if it's red, blue, purple,
    /// etc.
    #[palette(unsafe_same_layout_as = "T")]
    pub hue: LabHue<T>,

    /// The white point associated with the color's illuminant and observer.
    /// D65 for 2 degree observer is used by default.
    #[cfg_attr(feature = "serializing", serde(skip))]
    #[palette(unsafe_zero_sized)]
    pub white_point: PhantomData<Wp>,
}

impl<Wp, T> Copy for Lch<Wp, T> where T: Copy {}

impl<Wp, T> Clone for Lch<Wp, T>
where
    T: Clone,
{
    fn clone(&self) -> Lch<Wp, T> {
        Lch {
            l: self.l.clone(),
            chroma: self.chroma.clone(),
            hue: self.hue.clone(),
            white_point: PhantomData,
        }
    }
}

impl<Wp, T> Lch<Wp, T> {
    /// Create a CIE L\*C\*h° color.
    pub fn new<H: Into<LabHue<T>>>(l: T, chroma: T, hue: H) -> Self {
        Self::new_const(l, chroma, hue.into())
    }

    /// Create a CIE L\*C\*h° color. This is the same as `Lch::new` without the
    /// generic hue type. It's temporary until `const fn` supports traits.
    pub const fn new_const(l: T, chroma: T, hue: LabHue<T>) -> Self {
        Lch {
            l,
            chroma,
            hue,
            white_point: PhantomData,
        }
    }

    /// Convert to a `(L\*, C\*, h°)` tuple.
    pub fn into_components(self) -> (T, T, LabHue<T>) {
        (self.l, self.chroma, self.hue)
    }

    /// Convert from a `(L\*, C\*, h°)` tuple.
    pub fn from_components<H: Into<LabHue<T>>>((l, chroma, hue): (T, T, H)) -> Self {
        Self::new(l, chroma, hue)
    }
}

impl<Wp, T> Lch<Wp, T>
where
    T: Zero + Real,
{
    /// Return the `l` value minimum.
    pub fn min_l() -> T {
        T::zero()
    }

    /// Return the `l` value maximum.
    pub fn max_l() -> T {
        T::from_f64(100.0)
    }

    /// Return the `chroma` value minimum.
    pub fn min_chroma() -> T {
        T::zero()
    }

    /// Return the `chroma` value maximum. This value does not cover the entire
    /// color space, but covers enough to be practical for downsampling to
    /// smaller color spaces like sRGB.
    pub fn max_chroma() -> T {
        T::from_f64(128.0)
    }

    /// Return the `chroma` extended maximum value. This value covers the entire
    /// color space and is included for completeness, but the additional range
    /// should be unnecessary for most use cases.
    pub fn max_extended_chroma() -> T {
        T::from_f64(crate::num::Sqrt::sqrt(128.0f64 * 128.0 + 128.0 * 128.0))
    }
}

///<span id="Lcha"></span>[`Lcha`](crate::Lcha) implementations.
impl<Wp, T, A> Alpha<Lch<Wp, T>, A> {
    /// Create a CIE L\*C\*h° color with transparency.
    pub fn new<H: Into<LabHue<T>>>(l: T, chroma: T, hue: H, alpha: A) -> Self {
        Self::new_const(l, chroma, hue.into(), alpha)
    }

    /// Create a CIE L\*C\*h° color with transparency. This is the same as
    /// `Lcha::new` without the generic hue type. It's temporary until `const
    /// fn` supports traits.
    pub const fn new_const(l: T, chroma: T, hue: LabHue<T>, alpha: A) -> Self {
        Alpha {
            color: Lch::new_const(l, chroma, hue),
            alpha,
        }
    }

    /// Convert to a `(L\*, C\*, h°, alpha)` tuple.
    pub fn into_components(self) -> (T, T, LabHue<T>, A) {
        (self.color.l, self.color.chroma, self.color.hue, self.alpha)
    }

    /// Convert from a `(L\*, C\*, h°, alpha)` tuple.
    pub fn from_components<H: Into<LabHue<T>>>((l, chroma, hue, alpha): (T, T, H, A)) -> Self {
        Self::new(l, chroma, hue, alpha)
    }
}

impl<Wp, T> FromColorUnclamped<Lch<Wp, T>> for Lch<Wp, T> {
    fn from_color_unclamped(color: Lch<Wp, T>) -> Self {
        color
    }
}

impl<Wp, T> FromColorUnclamped<Lab<Wp, T>> for Lch<Wp, T>
where
    T: Zero + Hypot,
    Lab<Wp, T>: GetHue<Hue = LabHue<T>>,
{
    fn from_color_unclamped(color: Lab<Wp, T>) -> Self {
        Lch {
            hue: color.get_hue().unwrap_or_else(|| LabHue::from(T::zero())),
            l: color.l,
            chroma: color.a.hypot(color.b),
            white_point: PhantomData,
        }
    }
}

impl<Wp, T, H: Into<LabHue<T>>> From<(T, T, H)> for Lch<Wp, T> {
    fn from(components: (T, T, H)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp, T> From<Lch<Wp, T>> for (T, T, LabHue<T>) {
    fn from(color: Lch<Wp, T>) -> (T, T, LabHue<T>) {
        color.into_components()
    }
}

impl<Wp, T, H: Into<LabHue<T>>, A> From<(T, T, H, A)> for Alpha<Lch<Wp, T>, A> {
    fn from(components: (T, T, H, A)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp, T, A> From<Alpha<Lch<Wp, T>, A>> for (T, T, LabHue<T>, A) {
    fn from(color: Alpha<Lch<Wp, T>, A>) -> (T, T, LabHue<T>, A) {
        color.into_components()
    }
}

impl<Wp, T> IsWithinBounds for Lch<Wp, T>
where
    T: Zero + Real + PartialOrd,
{
    #[inline]
    fn is_within_bounds(&self) -> bool {
        self.l >= Self::min_l() && self.l <= Self::max_l() && self.chroma >= Self::min_chroma()
    }
}

impl<Wp, T> Clamp for Lch<Wp, T>
where
    T: Zero + Real + PartialOrd,
{
    #[inline]
    fn clamp(self) -> Self {
        Self::new(
            clamp(self.l, Self::min_l(), Self::max_l()),
            clamp_min(self.chroma, Self::min_chroma()),
            self.hue,
        )
    }
}

impl<Wp, T> ClampAssign for Lch<Wp, T>
where
    T: Zero + Real + PartialOrd,
{
    #[inline]
    fn clamp_assign(&mut self) {
        clamp_assign(&mut self.l, Self::min_l(), Self::max_l());
        clamp_min_assign(&mut self.chroma, Self::min_chroma());
    }
}

impl_mix_hue!(Lch<Wp> {l, chroma} phantom: white_point);
impl_lighten!(Lch<Wp> increase {l => [Self::min_l(), Self::max_l()]} other {hue, chroma} phantom: white_point);
impl_saturate!(Lch<Wp> increase {chroma => [Self::min_chroma(), Self::max_chroma()]} other {hue, l} phantom: white_point);

impl<Wp, T> GetHue for Lch<Wp, T>
where
    T: Zero + PartialOrd + Clone,
{
    type Hue = LabHue<T>;

    #[inline]
    fn get_hue(&self) -> Option<LabHue<T>> {
        if self.chroma <= T::zero() {
            None
        } else {
            Some(self.hue.clone())
        }
    }
}

impl<Wp, T, H> WithHue<H> for Lch<Wp, T>
where
    H: Into<LabHue<T>>,
{
    #[inline]
    fn with_hue(mut self, hue: H) -> Self {
        self.hue = hue.into();
        self
    }
}

impl<Wp, T, H> SetHue<H> for Lch<Wp, T>
where
    H: Into<LabHue<T>>,
{
    #[inline]
    fn set_hue(&mut self, hue: H) {
        self.hue = hue.into();
    }
}

impl<Wp, T> ShiftHue for Lch<Wp, T>
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

impl<Wp, T> ShiftHueAssign for Lch<Wp, T>
where
    T: AddAssign,
{
    type Scalar = T;

    #[inline]
    fn shift_hue_assign(&mut self, amount: Self::Scalar) {
        self.hue += amount;
    }
}

/// CIEDE2000 distance metric for color difference.
impl<Wp, T> ColorDifference for Lch<Wp, T>
where
    T: Real
        + RealAngle
        + One
        + Zero
        + Trigonometry
        + Abs
        + Sqrt
        + Powi
        + Exp
        + Arithmetics
        + PartialOrd
        + Clone,
    Self: Into<LabColorDiff<T>>,
{
    type Scalar = T;

    #[inline]
    fn get_color_difference(self, other: Lch<Wp, T>) -> Self::Scalar {
        get_ciede_difference(self.into(), other.into())
    }
}

impl<Wp, T> Default for Lch<Wp, T>
where
    T: Zero + Real,
    LabHue<T>: Default,
{
    fn default() -> Lch<Wp, T> {
        Lch::new(Self::min_l(), Self::min_chroma(), LabHue::default())
    }
}

impl_color_add!(Lch<Wp, T>, [l, chroma, hue], white_point);
impl_color_sub!(Lch<Wp, T>, [l, chroma, hue], white_point);

impl_array_casts!(Lch<Wp, T>, [T; 3]);

impl_eq_hue!(Lch<Wp>, LabHue, [l, chroma, hue]);

impl<Wp, T> RelativeContrast for Lch<Wp, T>
where
    T: Real + Arithmetics + PartialOrd,
    Xyz<Wp, T>: FromColor<Self>,
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
impl<Wp, T> Distribution<Lch<Wp, T>> for Standard
where
    T: Real + Zero + Sqrt + core::ops::Mul<Output = T>,
    Standard: Distribution<T> + Distribution<LabHue<T>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Lch<Wp, T> {
        Lch {
            l: rng.gen::<T>() * Lch::<Wp, T>::max_l(),
            chroma: rng.gen::<T>().sqrt() * Lch::<Wp, T>::max_chroma(),
            hue: rng.gen::<LabHue<T>>(),
            white_point: PhantomData,
        }
    }
}

#[cfg(feature = "random")]
pub struct UniformLch<Wp, T>
where
    T: SampleUniform,
{
    l: Uniform<T>,
    chroma: Uniform<T>,
    hue: crate::hues::UniformLabHue<T>,
    white_point: PhantomData<Wp>,
}

#[cfg(feature = "random")]
impl<Wp, T> SampleUniform for Lch<Wp, T>
where
    T: Sqrt + core::ops::Mul<Output = T> + Clone + SampleUniform,
    LabHue<T>: SampleBorrow<LabHue<T>>,
    crate::hues::UniformLabHue<T>: UniformSampler<X = LabHue<T>>,
{
    type Sampler = UniformLch<Wp, T>;
}

#[cfg(feature = "random")]
impl<Wp, T> UniformSampler for UniformLch<Wp, T>
where
    T: Sqrt + core::ops::Mul<Output = T> + Clone + SampleUniform,
    LabHue<T>: SampleBorrow<LabHue<T>>,
    crate::hues::UniformLabHue<T>: UniformSampler<X = LabHue<T>>,
{
    type X = Lch<Wp, T>;

    fn new<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        UniformLch {
            l: Uniform::new::<_, T>(low.l, high.l),
            chroma: Uniform::new::<_, T>(
                low.chroma.clone() * low.chroma,
                high.chroma.clone() * high.chroma,
            ),
            hue: crate::hues::UniformLabHue::new(low.hue, high.hue),
            white_point: PhantomData,
        }
    }

    fn new_inclusive<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        UniformLch {
            l: Uniform::new_inclusive::<_, T>(low.l, high.l),
            chroma: Uniform::new_inclusive::<_, T>(
                low.chroma.clone() * low.chroma,
                high.chroma.clone() * high.chroma,
            ),
            hue: crate::hues::UniformLabHue::new_inclusive(low.hue, high.hue),
            white_point: PhantomData,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Lch<Wp, T> {
        Lch {
            l: self.l.sample(rng),
            chroma: self.chroma.sample(rng).sqrt(),
            hue: self.hue.sample(rng),
            white_point: PhantomData,
        }
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<Wp, T> bytemuck::Zeroable for Lch<Wp, T> where T: bytemuck::Zeroable {}

#[cfg(feature = "bytemuck")]
unsafe impl<Wp: 'static, T> bytemuck::Pod for Lch<Wp, T> where T: bytemuck::Pod {}

#[cfg(test)]
mod test {
    use crate::white_point::D65;
    use crate::Lch;

    #[test]
    fn ranges() {
        assert_ranges! {
            Lch<D65, f64>;
            clamped {
                l: 0.0 => 100.0
            }
            clamped_min {
                chroma: 0.0 => 200.0
            }
            unclamped {
                hue: -360.0 => 360.0
            }
        }
    }

    raw_pixel_conversion_tests!(Lch<D65>: l, chroma, hue);
    raw_pixel_conversion_fail_tests!(Lch<D65>: l, chroma, hue);

    #[test]
    fn check_min_max_components() {
        assert_relative_eq!(Lch::<D65, f32>::min_l(), 0.0);
        assert_relative_eq!(Lch::<D65, f32>::max_l(), 100.0);
        assert_relative_eq!(Lch::<D65, f32>::min_chroma(), 0.0);
        assert_relative_eq!(Lch::<D65, f32>::max_chroma(), 128.0);
        assert_relative_eq!(Lch::<D65, f32>::max_extended_chroma(), 181.01933598375618);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn serialize() {
        let serialized = ::serde_json::to_string(&Lch::<D65>::new(0.3, 0.8, 0.1)).unwrap();

        assert_eq!(serialized, r#"{"l":0.3,"chroma":0.8,"hue":0.1}"#);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn deserialize() {
        let deserialized: Lch =
            ::serde_json::from_str(r#"{"l":0.3,"chroma":0.8,"hue":0.1}"#).unwrap();

        assert_eq!(deserialized, Lch::new(0.3, 0.8, 0.1));
    }

    #[cfg(feature = "random")]
    test_uniform_distribution! {
        Lch<D65, f32> as crate::Lab {
            l: (0.0, 100.0),
            a: (-89.0, 89.0),
            b: (-89.0, 89.0),
        },
        min: Lch::new(0.0f32, 0.0, 0.0),
        max: Lch::new(100.0, 128.0, 360.0)
    }
}
