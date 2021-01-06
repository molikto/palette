use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign};

#[cfg(feature = "random")]
use rand::distributions::uniform::{SampleBorrow, SampleUniform, Uniform, UniformSampler};
#[cfg(feature = "random")]
use rand::distributions::{Distribution, Standard};
#[cfg(feature = "random")]
use rand::Rng;

use crate::color_difference::ColorDifference;
use crate::color_difference::{get_ciede_difference, LabColorDiff};
use crate::convert::FromColorUnclamped;
use crate::encoding::pixel::RawPixel;
use crate::white_point::{WhitePoint, D65};
use crate::{
    clamp, contrast_ratio, from_f64, Alpha, Component, ComponentWise, FloatComponent, GetHue,
    Limited, Mix, OklabHue, Pixel, RelativeContrast, Shade, Xyz,
};

const M1: [[f64; 3]; 3] = [
    [0.8189330101, 0.3618667424, -0.1288597137],
    [0.0329845436, 0.9293118715, 0.0361456387],
    [0.0482003018, 0.2643662691, 0.6338517070],
];

pub(crate) const M1_INV: [[f64; 3]; 3] = [
    [1.2270138511, -0.5577999807, 0.2812561490],
    [-0.0405801784, 1.1122568696, -0.0716766787],
    [-0.0763812845, -0.4214819784, 1.5861632204],
];

const M2: [[f64; 3]; 3] = [
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
];

pub(crate) const M2_INV: [[f64; 3]; 3] = [
    [0.9999999985, 0.3963377922, 0.2158037581],
    [1.0000000089, -0.1055613423, -0.0638541748],
    [1.0000000547, -0.0894841821, -1.2914855379],
];

/// Oklab with an alpha component. See the [`Oklaba` implementation in
/// `Alpha`](crate::Alpha#Oklaba).
pub type Oklaba<Wp = D65, T = f32> = Alpha<Oklab<Wp, T>, T>;

/// The [Oklab color space](https://bottosson.github.io/posts/oklab/).
///
/// Oklab is a perceptually-uniform color space similar in structure to [L\*a\*b\*](crate::Lab), but
/// with better perceptual uniformity. It assumes a D65 whitepoint and normal well-lit viewing
/// conditions. As such, **the `Wp` type parameter is ignored** and is only present due to
/// complications with `FromColorUnclamped`.
#[derive(Debug, PartialEq, Pixel, FromColorUnclamped, WithAlpha)]
#[cfg_attr(feature = "serializing", derive(Serialize, Deserialize))]
#[palette(
    palette_internal,
    white_point = "Wp",
    component = "T",
    skip_derives(Oklab, Xyz)
)]
#[repr(C)]
pub struct Oklab<Wp = D65, T = f32>
where
    T: FloatComponent,
    Wp: WhitePoint,
{
    /// L is the lightness of the color. 0 gives absolute black and 1 gives the brightest white.
    pub l: T,

    /// a goes from red at -1 to green at 1.
    pub a: T,

    /// b goes from yellow at -1 to blue at 1.
    pub b: T,

    /// The white point associated with the color's illuminant and observer.
    /// D65 for 2 degree observer is used by default.
    #[cfg_attr(feature = "serializing", serde(skip))]
    #[palette(unsafe_zero_sized)]
    pub white_point: PhantomData<Wp>,
}

impl<Wp, T> Copy for Oklab<Wp, T>
where
    T: FloatComponent,
    Wp: WhitePoint,
{
}

impl<Wp, T> Clone for Oklab<Wp, T>
where
    T: FloatComponent,
    Wp: WhitePoint,
{
    fn clone(&self) -> Oklab<Wp, T> {
        *self
    }
}

impl<Wp, T> Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    /// Create an Oklab color.
    pub fn new(l: T, a: T, b: T) -> Self {
        Self {
            l,
            a,
            b,
            white_point: PhantomData,
        }
    }

    /// Convert to a `(L, a, b)` tuple.
    pub fn into_components(self) -> (T, T, T) {
        (self.l, self.a, self.b)
    }

    /// Convert from a `(L, a, b)` tuple.
    pub fn from_components((l, a, b): (T, T, T)) -> Self {
        Self::new(l, a, b)
    }

    /// Return the `l` value minimum.
    pub fn min_l() -> T {
        from_f64(0.0)
    }

    /// Return the `l` value maximum.
    pub fn max_l() -> T {
        from_f64(1.0)
    }

    /// Return the `a` value minimum.
    pub fn min_a() -> T {
        from_f64(-1.0)
    }

    /// Return the `a` value maximum.
    pub fn max_a() -> T {
        from_f64(1.0)
    }

    /// Return the `b` value minimum.
    pub fn min_b() -> T {
        from_f64(-1.0)
    }

    /// Return the `b` value maximum.
    pub fn max_b() -> T {
        from_f64(1.0)
    }
}

///<span id="Oklaba"></span>[`Oklaba`](crate::Oklaba) implementations.
impl<Wp, T, A> Alpha<Oklab<Wp, T>, A>
where
    Wp: WhitePoint,
    T: FloatComponent,
    A: Component,
{
    /// Oklab and transparency.
    pub fn new(l: T, a: T, b: T, alpha: A) -> Self {
        Alpha {
            color: Oklab::new(l, a, b),
            alpha,
        }
    }

    /// Convert to a `(L, a, b, alpha)` tuple.
    pub fn into_components(self) -> (T, T, T, A) {
        (self.l, self.a, self.b, self.alpha)
    }

    /// Convert from a `(L, a, b, alpha)` tuple.
    pub fn from_components((l, a, b, alpha): (T, T, T, A)) -> Self {
        Self::new(l, a, b, alpha)
    }
}

impl<Wp, T> FromColorUnclamped<Oklab<Wp, T>> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    fn from_color_unclamped(color: Self) -> Self {
        color
    }
}

/// This implementation is **only valid if the XYZ white point is D65**.
impl<Wp, T> FromColorUnclamped<Xyz<Wp, T>> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    fn from_color_unclamped(color: Xyz<Wp, T>) -> Self {
        let l = from_f64::<T>(M1[0][0]) * color.x
            + from_f64::<T>(M1[0][1]) * color.y
            + from_f64::<T>(M1[0][2]) * color.z;

        let m = from_f64::<T>(M1[1][0]) * color.x
            + from_f64::<T>(M1[1][1]) * color.y
            + from_f64::<T>(M1[1][2]) * color.z;

        let s = from_f64::<T>(M1[2][0]) * color.x
            + from_f64::<T>(M1[2][1]) * color.y
            + from_f64::<T>(M1[2][2]) * color.z;

        let l_ = l.cbrt();
        let m_ = m.cbrt();
        let s_ = s.cbrt();

        let l = from_f64::<T>(M2[0][0]) * l_
            + from_f64::<T>(M2[0][1]) * m_
            + from_f64::<T>(M2[0][2]) * s_;

        let a = from_f64::<T>(M2[1][0]) * l_
            + from_f64::<T>(M2[1][1]) * m_
            + from_f64::<T>(M2[1][2]) * s_;

        let b = from_f64::<T>(M2[2][0]) * l_
            + from_f64::<T>(M2[2][1]) * m_
            + from_f64::<T>(M2[2][2]) * s_;

        Self::new(l, a, b)
    }
}

impl<Wp, T: FloatComponent> From<(T, T, T)> for Oklab<Wp, T>
where
    Wp: WhitePoint,
{
    fn from(components: (T, T, T)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp, T: FloatComponent> Into<(T, T, T)> for Oklab<Wp, T>
where
    Wp: WhitePoint,
{
    fn into(self) -> (T, T, T) {
        self.into_components()
    }
}

impl<Wp: WhitePoint, T: FloatComponent, A: Component> From<(T, T, T, A)>
    for Alpha<Oklab<Wp, T>, A>
{
    fn from(components: (T, T, T, A)) -> Self {
        Self::from_components(components)
    }
}

impl<Wp: WhitePoint, T: FloatComponent, A: Component> Into<(T, T, T, A)>
    for Alpha<Oklab<Wp, T>, A>
{
    fn into(self) -> (T, T, T, A) {
        self.into_components()
    }
}

impl<Wp, T> Limited for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    #[rustfmt::skip]
    fn is_valid(&self) -> bool {
        self.l >= from_f64(0.0) && self.l <= from_f64(1.0) &&
        self.a >= from_f64(-1.0) && self.a <= from_f64(1.0) &&
        self.b >= from_f64(-1.0) && self.b <= from_f64(1.0)
    }

    fn clamp(&self) -> Self {
        let mut c = *self;
        c.clamp_self();
        c
    }

    fn clamp_self(&mut self) {
        self.l = clamp(self.l, from_f64(0.0), from_f64(1.0));
        self.a = clamp(self.a, from_f64(-1.0), from_f64(1.0));
        self.b = clamp(self.b, from_f64(-1.0), from_f64(1.0));
    }
}

impl<Wp, T> Mix for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Scalar = T;

    fn mix(&self, other: &Self, factor: T) -> Self {
        let factor = clamp(factor, T::zero(), T::one());

        Self::new(
            self.l + factor * (other.l - self.l),
            self.a + factor * (other.a - self.a),
            self.b + factor * (other.b - self.b),
        )
    }
}

impl<Wp, T> Shade for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Scalar = T;

    fn lighten(&self, amount: T) -> Self {
        Self::new(self.l + amount * from_f64(1.0), self.a, self.b)
    }
}

impl<Wp, T> GetHue for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Hue = OklabHue<T>;

    fn get_hue(&self) -> Option<OklabHue<T>> {
        if self.a == T::zero() && self.b == T::zero() {
            None
        } else {
            Some(OklabHue::from_radians(self.b.atan2(self.a)))
        }
    }
}

impl<Wp, T> ColorDifference for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Scalar = T;

    fn get_color_difference(&self, other: &Self) -> Self::Scalar {
        // Color difference calculation requires Lab and chroma components. This
        // function handles the conversion into those components which are then
        // passed to `get_ciede_difference()` where calculation is completed.
        let self_params = LabColorDiff {
            l: self.l,
            a: self.a,
            b: self.b,
            chroma: (self.a * self.a + self.b * self.b).sqrt(),
        };
        let other_params = LabColorDiff {
            l: other.l,
            a: other.a,
            b: other.b,
            chroma: (other.a * other.a + other.b * other.b).sqrt(),
        };

        get_ciede_difference(&self_params, &other_params)
    }
}

impl<Wp, T> ComponentWise for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Scalar = T;

    fn component_wise<F: FnMut(T, T) -> T>(&self, other: &Self, mut f: F) -> Self {
        Self::new(f(self.l, other.l), f(self.a, other.a), f(self.b, other.b))
    }

    fn component_wise_self<F: FnMut(T) -> T>(&self, mut f: F) -> Self {
        Self::new(f(self.l), f(self.a), f(self.b))
    }
}

impl<Wp, T> Default for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    fn default() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
}

impl<Wp, T> Add for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn add(self, other: Self) -> Self::Output {
        Self::new(self.l + other.l, self.a + other.a, self.b + other.b)
    }
}

impl<Wp, T> Add<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn add(self, c: T) -> Self::Output {
        Self::new(self.l + c, self.a + c, self.b + c)
    }
}

impl<Wp, T> AddAssign for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + AddAssign,
{
    fn add_assign(&mut self, other: Self) {
        self.l += other.l;
        self.a += other.a;
        self.b += other.b;
    }
}

impl<Wp, T> AddAssign<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + AddAssign,
{
    fn add_assign(&mut self, c: T) {
        self.l += c;
        self.a += c;
        self.b += c;
    }
}

impl<Wp, T> Sub for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        Self::new(self.l - other.l, self.a - other.a, self.b - other.b)
    }
}

impl<Wp, T> Sub<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn sub(self, c: T) -> Self::Output {
        Self::new(self.l - c, self.a - c, self.b - c)
    }
}

impl<Wp, T> SubAssign for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + SubAssign,
{
    fn sub_assign(&mut self, other: Self) {
        self.l -= other.l;
        self.a -= other.a;
        self.b -= other.b;
    }
}

impl<Wp, T> SubAssign<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + SubAssign,
{
    fn sub_assign(&mut self, c: T) {
        self.l -= c;
        self.a -= c;
        self.b -= c;
    }
}

impl<Wp, T> Mul for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn mul(self, other: Self) -> Self::Output {
        Self::new(self.l * other.l, self.a * other.a, self.b * other.b)
    }
}

impl<Wp, T> Mul<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn mul(self, c: T) -> Self::Output {
        Self::new(self.l * c, self.a * c, self.b * c)
    }
}

impl<Wp, T> MulAssign for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + MulAssign,
{
    fn mul_assign(&mut self, other: Self) {
        self.l *= other.l;
        self.a *= other.a;
        self.b *= other.b;
    }
}

impl<Wp, T> MulAssign<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + MulAssign,
{
    fn mul_assign(&mut self, c: T) {
        self.l *= c;
        self.a *= c;
        self.b *= c;
    }
}

impl<Wp, T> Div for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn div(self, other: Self) -> Self::Output {
        Self::new(self.l / other.l, self.a / other.a, self.b / other.b)
    }
}

impl<Wp, T> Div<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Output = Self;

    fn div(self, c: T) -> Self::Output {
        Self::new(self.l / c, self.a / c, self.b / c)
    }
}

impl<Wp, T> DivAssign for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + DivAssign,
{
    fn div_assign(&mut self, other: Self) {
        self.l /= other.l;
        self.a /= other.a;
        self.b /= other.b;
    }
}

impl<Wp, T> DivAssign<T> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + DivAssign,
{
    fn div_assign(&mut self, c: T) {
        self.l /= c;
        self.a /= c;
        self.b /= c;
    }
}

impl<Wp, T, P> AsRef<P> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
    P: RawPixel<T> + ?Sized,
{
    fn as_ref(&self) -> &P {
        self.as_raw()
    }
}

impl<Wp, T, P> AsMut<P> for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
    P: RawPixel<T> + ?Sized,
{
    fn as_mut(&mut self) -> &mut P {
        self.as_raw_mut()
    }
}

impl<Wp, T> RelativeContrast for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent,
{
    type Scalar = T;

    fn get_contrast_ratio(&self, other: &Self) -> T {
        use crate::FromColor;

        let xyz1 = Xyz::from_color(*self);
        let xyz2 = Xyz::from_color(*other);

        contrast_ratio(xyz1.y, xyz2.y)
    }
}

#[cfg(feature = "random")]
impl<Wp, T> Distribution<Oklab<Wp, T>> for Standard
where
    Wp: WhitePoint,
    T: FloatComponent,
    Standard: Distribution<T>,
{
    // `a` and `b` both range from (-1.0, 1.0)
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Oklab<Wp, T>
    where
        Wp: WhitePoint,
    {
        Oklab::new(
            rng.gen() * from_f64(1.0),
            rng.gen() * from_f64(2.0) - from_f64(1.0),
            rng.gen() * from_f64(2.0) - from_f64(1.0),
        )
    }
}

#[cfg(feature = "random")]
pub struct UniformOklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + SampleUniform,
{
    l: Uniform<T>,
    a: Uniform<T>,
    b: Uniform<T>,
    white_point: PhantomData<Wp>,
}

#[cfg(feature = "random")]
impl<Wp, T> SampleUniform for Oklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + SampleUniform,
{
    type Sampler = UniformOklab<Wp, T>;
}

#[cfg(feature = "random")]
impl<Wp, T> UniformSampler for UniformOklab<Wp, T>
where
    Wp: WhitePoint,
    T: FloatComponent + SampleUniform,
{
    type X = Oklab<Wp, T>;

    fn new<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low = *low_b.borrow();
        let high = *high_b.borrow();

        Self {
            l: Uniform::new::<_, T>(low.l, high.l),
            a: Uniform::new::<_, T>(low.a, high.a),
            b: Uniform::new::<_, T>(low.b, high.b),
            white_point: PhantomData,
        }
    }

    fn new_inclusive<B1, B2>(low_b: B1, high_b: B2) -> Self
    where
        B1: SampleBorrow<Self::X> + Sized,
        B2: SampleBorrow<Self::X> + Sized,
    {
        let low = *low_b.borrow();
        let high = *high_b.borrow();

        Self {
            l: Uniform::new_inclusive::<_, T>(low.l, high.l),
            a: Uniform::new_inclusive::<_, T>(low.a, high.a),
            b: Uniform::new_inclusive::<_, T>(low.b, high.b),
            white_point: PhantomData,
        }
    }

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Oklab<Wp, T>
    where
        Wp: WhitePoint,
    {
        Oklab::new(self.l.sample(rng), self.a.sample(rng), self.b.sample(rng))
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{FromColor, LinSrgb};

    #[test]
    fn red() {
        let a = Oklab::from_color(LinSrgb::new(1.0, 0.0, 0.0));
        let b = Oklab::new(0.627986, 0.224840, 0.125798);
        assert_relative_eq!(a, b, epsilon = 0.00001);
    }

    #[test]
    fn green() {
        let a = Oklab::from_color(LinSrgb::new(0.0, 1.0, 0.0));
        let b = Oklab::new(0.866432, -0.233916, 0.179417);
        assert_relative_eq!(a, b, epsilon = 0.00001);
    }

    #[test]
    fn blue() {
        let a = Oklab::from_color(LinSrgb::new(0.0, 0.0, 1.0));
        let b = Oklab::new(0.451977, -0.032429, -0.311611);
        assert_relative_eq!(a, b, epsilon = 0.00001);
    }

    #[test]
    fn ranges() {
        assert_ranges! {
            Oklab<D65, f64>;
            limited {
                l: 0.0 => 1.0,
                a: -1.0 => 1.0,
                b: -1.0 => 1.0
            }
            limited_min {}
            unlimited {}
        }
    }

    #[test]
    fn check_min_max_components() {
        assert_relative_eq!(Oklab::<D65>::min_l(), 0.0);
        assert_relative_eq!(Oklab::<D65>::min_a(), -1.0);
        assert_relative_eq!(Oklab::<D65>::min_b(), -1.0);
        assert_relative_eq!(Oklab::<D65>::max_l(), 1.0);
        assert_relative_eq!(Oklab::<D65>::max_a(), 1.0);
        assert_relative_eq!(Oklab::<D65>::max_b(), 1.0);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn serialize() {
        let serialized = ::serde_json::to_string(&Oklab::<D65>::new(0.3, 0.8, 0.1)).unwrap();

        assert_eq!(serialized, r#"{"l":0.3,"a":0.8,"b":0.1}"#);
    }

    #[cfg(feature = "serializing")]
    #[test]
    fn deserialize() {
        let deserialized: Oklab = ::serde_json::from_str(r#"{"l":0.3,"a":0.8,"b":0.1}"#).unwrap();

        assert_eq!(deserialized, Oklab::new(0.3, 0.8, 0.1));
    }

    #[cfg(feature = "random")]
    test_uniform_distribution! {
        Oklab {
            l: (0.0, 1.0),
            a: (-1.0, 1.0),
            b: (-1.0, 1.0)
        },
        min: Oklab::new(0.0, -1.0, -1.0),
        max: Oklab::new(1.0, 1.0, 1.0)
    }
}
