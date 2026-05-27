use ferrum_gpu_core::{Dim3, Direction};

#[test]
fn dim3_constructor_and_components() {
    let d = Dim3::new(16, 8, 4);
    assert_eq!(d.x, 16);
    assert_eq!(d.y, 8);
    assert_eq!(d.z, 4);
}

#[test]
fn dim3_from_tuple() {
    let d: Dim3 = (32, 1, 1).into();
    assert_eq!(d, Dim3::new(32, 1, 1));
}

#[test]
fn dim3_default_is_unit() {
    let d = Dim3::default();
    assert_eq!(d, Dim3::new(1, 1, 1));
}

#[test]
fn direction_inverse_negates_sign() {
    assert_eq!(Direction::Forward.sign(), 1);
    assert_eq!(Direction::Inverse.sign(), -1);
}
