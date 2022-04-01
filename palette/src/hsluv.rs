use core::{
    marker::PhantomData,
    ops::{Add, AddAssign, Sub, SubAssign},
};

#[cfg(feature = "random")]
use rand::{
    distributions::{
        uniform::{SampleBorrow, SampleUniform, Uniform, UniformSampler},
        Distribution, Standard,
    },
    Rng,
};

use approx::{AbsDiffEq, RelativeEq, UlpsEq};

#[cfg(feature = "random")]
use crate::num::{Cbrt, Sqrt};

use crate::{
    angle::{RealAngle, SignedAngle},
    clamp, clamp_assign, contrast_ratio,
    convert::FromColorUnclamped,
    luv_bounds::LuvBounds,
    num::{Arithmetics, MinMax, One, Powi, Real, Zero},
    white_point::D65,
    Alpha, Clamp, ClampAssign, FromColor, GetHue, IsWithinBounds, Lchuv, Lighten, LightenAssign,
    LuvHue, Mix, MixAssign, RelativeContrast, Saturate, SaturateAssign, SetHue, ShiftHue,
    ShiftHueAssign, WithHue, Xyz,
};

/// HSLuv with an alpha component. See the [`Hsluva` implementation in
/// `Alpha`](crate::Alpha#Hsluva).
pub type Hsluva<Wp = D65, T = f32> = Alpha<Hsluv<Wp, T>, T>;

/// HSLuv color space.
///
/// The HSLuv color space can be seen as a cylindrical version of
/// [CIELUV](crate::luv::Luv), similar to
/// [LCHuv](crate::lchuv::Lchuv), with the additional benefit of
/// streching the chroma values to a uniform saturation range [0.0,
/// 100.0]. This makes HSLuv much more convenient for generating
/// colors than Lchuv, as the set of valid saturation values is
/// independent of lightness and hue.
#[derive(ArrayCast, FromColorUnclamped, WithAlpha)]
#[cfg_attr(feature = "serializing", derive(Serialize, Deserialize))]
#[palette(
    palette_internal,
    white_point = "Wp",
    component = "T",
    skip_derives(Lchuv, Hsluv)
)]
#[repr(C)]
pub struct Hsluv<Wp = D65, T = f32> {
    /// The hue of the color, in degrees. Decides if it's red, blue, purple,
    /// etc.
    #[palette(unsafe_same_layout_as = "T")]
    pub hue: LuvHue<T>,

    /// The colorfulness of the color, as a percentage of the maximum
    /// available chroma. 0.0 gives gray scale colors and 100.0 will
    /// give absolutely clear colors.
    pub saturation: T,

    /// Decides how light the color will look. 0.0 will be black, 50.0 will give
    /// a clear color, and 100.0 will give white.
    pub l: T,

    /// The white point and RGB primaries this color is adapted to. The default
    /// is the sRGB standard.
    #[cfg_attr(feature = "serializing", serde(skip))]
    #[palette(unsafe_zero_sized)]
    pub white_point: PhantomData<Wp>,
}

impl<Wp, T> Copy for Hsluv<Wp, T> where T: Copy {}

impl<Wp, T> Clone for Hsluv<Wp, T>
where
    T: Clone,
{
    fn clone(&self) -> Hsluv<Wp, T> {
        Hsluv {
            hue: self.hue.clone(),
            saturation: self.saturation.clone(),
            l: self.l.clone(),
            white_point: PhantomData,
        }
    }
}

impl<Wp, T> Hsluv<Wp, T> {
    /// Create an HSLuv color.
    pub fn new<H: Into<LuvHue<T>>>(hue: H, saturation: T, l: T) -> Self {
        Self::new_const(hue.into(), saturation, l)
    }

    /// Create an HSLuv color. This is the same as `Hsluv::new` without the
    /// generic hue type. It's temporary until `const fn` supports traits.
    pub const fn new_const(hue: LuvHue<T>, saturation: T, l: T) -> Self {
        Hsluv {
            hue,
            saturation,
            l,
            white_point: PhantomData,
        }
    }

    /// Convert to a `(hue, saturation, l)` tuple.
    pub fn into_components(self) -> (LuvHue<T>, T, T) {
        (self.hue, self.saturation, self.l)
    }

    /// Convert from a `(hue, saturation, l)` tuple.
    pub fn from_components<H: Into<LuvHue<T>>>((hue, saturation, l): (H, T, T)) -> Self {
        Self::new(hue, saturation, l)
    }
}

impl<Wp, T> Hsluv<Wp, T>
where
    T: Zero + Real,
{
    /// Return the `saturation` value minimum.
    pub fn min_saturation() -> T {
        T::zero()
    }

    /// Return the `saturation` value maximum.
    pub fn max_saturation() -> T {
        T::from_f64(100.0)
    }

    /// Return the `l` value minimum.
    pub fn min_l() -> T {
        T::zero()
    }

    /// Return the `l` value maximum.
    pub fn max_l() -> T {
        T::from_f64(100.0)
    }
}

///<span id="Hsluva"></span>[`Hsluva`](crate::Hsluva) implementations.
impl<Wp, T, A> Alpha<Hsluv<Wp, T>, A> {
    /// Create an HSLuv color with transparency.
    pub fn new<H: Into<LuvHue<T>>>(hue: H, saturation: T, l: T, alpha: A) -> Self {
        Self::new_const(hue.into(), saturation, l, alpha)
    }

    /// Create an HSLuv color with transparency. This is the same as
    /// `Hsluva::new` without the generic hue type. It's temporary until `const
    /// fn` supports traits.
    pub const fn new_const(hue: LuvHue<T>, saturation: T, l: T, alpha: A) -> Self {
        Alpha {
            color: Hsluv::new_const(hue, saturation, l),
            alpha,
        }
    }

    /// Convert to a `(hue, saturation, l, alpha)` tuple.
    pub fn into_components(self) -> (LuvHue<T>, T, T, A) {
        (
            self.color.hue,
            self.color.saturation,
            self.color.l,
            self.alpha,
        )
    }

    /// Convert from a `(hue, saturation, l, alpha)` tuple.
    pub fn from_components<H: Into<LuvHue<T>>>((hue, saturation, l, alpha): (H, T, T, A)) -> Self {
        Self::new(hue, saturation, l, alpha)
    }
}

impl<Wp, T> FromColorUnclamped<Hsluv<Wp, T>> for Hsluv<Wp, T> {
    fn from_color_unclamped(hsluv: Hsluv<Wp, T>) -> Self {
        hsluv
    }
}

impl<Wp, T> FromColorUnclamped<Lchuv<Wp, T>> for Hsluv<Wp, T>
where
    T: Real + RealAngle + Into<f64> + Powi + Arithmetics + Clone,
{
    fn from_color_unclamped(color: Lchuv<Wp, T>) -> Self {
        // convert the chroma to a saturation based on the max
        // saturation at a particular hue.
        let max_chroma =
            LuvBounds::from_lightness(color.l.clone()).max_chroma_at_hue(color.hue.clone());

        Hsluv::new(
            color.hue,
            color.chroma / max_chroma * T::from_f64(100.0),
            color.l,
        )
    }
}

impl<Wp, T, H: Into<LuvHue<T>>> From<(H, T, T)> for Hsluv<Wp, T> {
    fn from(components: (H, T, T)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp, T> From<Hsluv<Wp, T>> for (LuvHue<T>, T, T) {
    fn from(color: Hsluv<Wp, T>) -> (LuvHue<T>, T, T) {
        color.into_components()
    }
}

impl<Wp, T, H: Into<LuvHue<T>>, A> From<(H, T, T, A)> for Alpha<Hsluv<Wp, T>, A> {
    fn from(components: (H, T, T, A)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp, T, A> From<Alpha<Hsluv<Wp, T>, A>> for (LuvHue<T>, T, T, A) {
    fn from(color: Alpha<Hsluv<Wp, T>, A>) -> (LuvHue<T>, T, T, A) {
        color.into_components()
    }
}

impl<Wp, T> IsWithinBounds for Hsluv<Wp, T>
where
    T: Zero + Real + PartialOrd,
{
    #[rustfmt::skip]
    #[inline]
    fn is_within_bounds(&self) -> bool {
        self.saturation >= Self::min_saturation() && self.saturation <= Self::max_saturation() &&
        self.l >= Self::min_l() && self.l <= Self::max_l()
    }
}

impl<Wp, T> Clamp for Hsluv<Wp, T>
where
    T: Zero + Real + PartialOrd,
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
            clamp(self.l, Self::min_l(), Self::max_l()),
        )
    }
}

impl<Wp, T> ClampAssign for Hsluv<Wp, T>
where
    T: Zero + Real + PartialOrd,
{
    #[inline]
    fn clamp_assign(&mut self) {
        clamp_assign(
            &mut self.saturation,
            Self::min_saturation(),
            Self::max_saturation(),
        );
        clamp_assign(&mut self.l, Self::min_l(), Self::max_l());
    }
}

impl_mix_hue!(Hsluv<Wp> {saturation, l} phantom: white_point);
impl_lighten!(Hsluv<Wp> increase {l => [Self::min_l(), Self::max_l()]} other {hue, saturation} phantom: white_point);
impl_saturate!(Hsluv<Wp> increase {saturation => [Self::min_saturation(), Self::max_saturation()]} other {hue, l} phantom: white_point);

impl<Wp, T> GetHue for Hsluv<Wp, T>
where
    T: Zero + PartialOrd + Clone,
{
    type Hue = LuvHue<T>;

    #[inline]
    fn get_hue(&self) -> Option<LuvHue<T>> {
        if self.saturation <= T::zero() {
            None
        } else {
            Some(self.hue.clone())
        }
    }
}

impl<Wp, T, H> WithHue<H> for Hsluv<Wp, T>
where
    H: Into<LuvHue<T>>,
{
    #[inline]
    fn with_hue(mut self, hue: H) -> Self {
        self.hue = hue.into();
        self
    }
}

impl<Wp, T, H> SetHue<H> for Hsluv<Wp, T>
where
    H: Into<LuvHue<T>>,
{
    #[inline]
    fn set_hue(&mut self, hue: H) {
        self.hue = hue.into();
    }
}

impl<Wp, T> ShiftHue for Hsluv<Wp, T>
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

impl<Wp, T> ShiftHueAssign for Hsluv<Wp, T>
where
    T: AddAssign,
{
    type Scalar = T;

    #[inline]
    fn shift_hue_assign(&mut self, amount: Self::Scalar) {
        self.hue += amount;
    }
}

impl<Wp, T> Default for Hsluv<Wp, T>
where
    T: Real + Zero,
    LuvHue<T>: Default,
{
    fn default() -> Hsluv<Wp, T> {
        Hsluv::new(LuvHue::default(), Self::min_saturation(), Self::min_l())
    }
}

impl_color_add!(Hsluv<Wp, T>, [hue, saturation, l], white_point);
impl_color_sub!(Hsluv<Wp, T>, [hue, saturation, l], white_point);

impl_array_casts!(Hsluv<Wp, T>, [T; 3]);

impl_eq_hue!(Hsluv<Wp>, LuvHue, [hue, saturation, l]);

impl<Wp, T> RelativeContrast for Hsluv<Wp, T>
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
impl<Wp, T> Distribution<Hsluv<Wp, T>> for Standard
where
    T: Real + Cbrt + Sqrt + Arithmetics + PartialOrd,
    Standard: Distribution<T> + Distribution<LuvHue<T>>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Hsluv<Wp, T> {
        crate::random_sampling::sample_hsluv(rng.gen::<LuvHue<T>>(), rng.gen(), rng.gen())
    }
}

#[cfg(feature = "random")]
pub struct UniformHsluv<Wp, T>
where
    T: SampleUniform,
{
    hue: crate::hues::UniformLuvHue<T>,
    u1: Uniform<T>,
    u2: Uniform<T>,
    space: PhantomData<Wp>,
}

#[cfg(feature = "random")]
impl<Wp, T> SampleUniform for Hsluv<Wp, T>
where
    T: Real + Cbrt + Sqrt + Powi + Arithmetics + PartialOrd + Clone + SampleUniform,
    LuvHue<T>: SampleBorrow<LuvHue<T>>,
    crate::hues::UniformLuvHue<T>: UniformSampler<X = LuvHue<T>>,
{
    type Sampler = UniformHsluv<Wp, T>;
}

#[cfg(feature = "random")]
impl<Wp, T> UniformSampler for UniformHsluv<Wp, T>
where
    T: Real + Cbrt + Sqrt + Powi + Arithmetics + PartialOrd + Clone + SampleUniform,
    LuvHue<T>: SampleBorrow<LuvHue<T>>,
    crate::hues::UniformLuvHue<T>: UniformSampler<X = LuvHue<T>>,
{
    type X = Hsluv<Wp, T>;

    fn new<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        use crate::random_sampling::invert_hsluv_sample;

        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        let (r1_min, r2_min): (T, T) = invert_hsluv_sample(low.saturation, low.l);
        let (r1_max, r2_max): (T, T) = invert_hsluv_sample(high.saturation, high.l);

        UniformHsluv {
            hue: crate::hues::UniformLuvHue::new(low.hue, high.hue),
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
        use crate::random_sampling::invert_hsluv_sample;

        let low = low_b.borrow().clone();
        let high = high_b.borrow().clone();

        let (r1_min, r2_min): (T, T) = invert_hsluv_sample(low.saturation, low.l);
        let (r1_max, r2_max): (T, T) = invert_hsluv_sample(high.saturation, high.l);

        UniformHsluv {
            hue: crate::hues::UniformLuvHue::new_inclusive(low.hue, high.hue),
            u1: Uniform::new_inclusive::<_, T>(r1_min, r1_max),
            u2: Uniform::new_inclusive::<_, T>(r2_min, r2_max),
            space: PhantomData,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Hsluv<Wp, T> {
        crate::random_sampling::sample_hsluv(
            self.hue.sample(rng),
            self.u1.sample(rng),
            self.u2.sample(rng),
        )
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<Wp, T> bytemuck::Zeroable for Hsluv<Wp, T> where T: bytemuck::Zeroable {}

#[cfg(feature = "bytemuck")]
unsafe impl<Wp: 'static, T> bytemuck::Pod for Hsluv<Wp, T> where T: bytemuck::Pod {}

#[cfg(test)]
mod test {
    use super::Hsluv;
    use crate::{white_point::D65, FromColor, Lchuv, LuvHue, Saturate};

    #[test]
    fn lchuv_round_trip() {
        for hue in (0..=20).map(|x| x as f64 * 18.0) {
            for sat in (0..=20).map(|x| x as f64 * 5.0) {
                for l in (1..=20).map(|x| x as f64 * 5.0) {
                    let hsluv = Hsluv::<D65, _>::new(hue, sat, l);
                    let lchuv = Lchuv::from_color(hsluv);
                    let mut to_hsluv = Hsluv::from_color(lchuv);
                    if to_hsluv.l < 1e-8 {
                        to_hsluv.hue = LuvHue::from(0.0);
                    }
                    assert_relative_eq!(hsluv, to_hsluv, epsilon = 1e-5);
                }
            }
        }
    }

    #[test]
    fn ranges() {
        assert_ranges! {
            Hsluv<D65, f64>;
            clamped {
                saturation: 0.0 => 100.0,
                l: 0.0 => 100.0
            }
            clamped_min {}
            unclamped {
                hue: -360.0 => 360.0
            }
        }
    }

    /// Check that the arithmetic operations (add/sub) are all
    /// implemented.
    #[test]
    fn test_arithmetic() {
        let hsl = Hsluv::<D65>::new(120.0, 40.0, 30.0);
        let hsl2 = Hsluv::new(200.0, 30.0, 40.0);
        let mut _hsl3 = hsl + hsl2;
        _hsl3 += hsl2;
        let mut _hsl4 = hsl2 + 0.3;
        _hsl4 += 0.1;

        _hsl3 = hsl2 - hsl;
        _hsl3 = _hsl4 - 0.1;
        _hsl4 -= _hsl3;
        _hsl3 -= 0.1;
    }

    #[test]
    fn saturate() {
        for sat in (0..=10).map(|s| s as f64 * 10.0) {
            for a in (0..=10).map(|l| l as f64 * 10.0) {
                let hsl = Hsluv::<D65, _>::new(150.0, sat, a);
                let hsl_sat_fixed = hsl.saturate_fixed(0.1);
                let expected_sat_fixed = Hsluv::new(150.0, (sat + 10.0).min(100.0), a);
                assert_relative_eq!(hsl_sat_fixed, expected_sat_fixed);

                let hsl_sat = hsl.saturate(0.1);
                let expected_sat = Hsluv::new(150.0, (sat + (100.0 - sat) * 0.1).min(100.0), a);
                assert_relative_eq!(hsl_sat, expected_sat);
            }
        }
    }

    raw_pixel_conversion_tests!(Hsluv<D65>: hue, saturation, lightness);
    raw_pixel_conversion_fail_tests!(Hsluv<D65>: hue, saturation, lightness);

    #[test]
    fn check_min_max_components() {
        assert_relative_eq!(Hsluv::<D65>::min_saturation(), 0.0);
        assert_relative_eq!(Hsluv::<D65>::min_l(), 0.0);
        assert_relative_eq!(Hsluv::<D65>::max_saturation(), 100.0);
        assert_relative_eq!(Hsluv::<D65>::max_l(), 100.0);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn serialize() {
        let serialized = ::serde_json::to_string(&Hsluv::<D65>::new(120.0, 80.0, 60.0)).unwrap();

        assert_eq!(serialized, r#"{"hue":120.0,"saturation":80.0,"l":60.0}"#);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn deserialize() {
        let deserialized: Hsluv =
            ::serde_json::from_str(r#"{"hue":120.0,"saturation":80.0,"l":60.0}"#).unwrap();

        assert_eq!(deserialized, Hsluv::new(120.0, 80.0, 60.0));
    }
}
