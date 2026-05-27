use ferrum_gpu_core::{
    ArgKind, BackendCaps, BackendId, KernelArtifact, KernelMeta, TargetInfo,
};

#[test]
fn artifact_constructor_smoke() {
    let art = KernelArtifact {
        backend: BackendId::Cuda,
        target: TargetInfo::Cuda { sm_major: 8, sm_minor: 0 },
        blob: vec![0xDE, 0xAD, 0xBE, 0xEF].into(),
        manifest: vec![KernelMeta {
            mangled_name: "vector_add".into(),
            arg_kinds: vec![ArgKind::BufferF32, ArgKind::BufferF32, ArgKind::BufferF32, ArgKind::ScalarU32],
            shared_memory_bytes: 0,
            recommended_block: None,
            required_caps: BackendCaps::Cuda { min_sm_major: 5, min_sm_minor: 0 },
        }],
    };
    assert_eq!(art.backend, BackendId::Cuda);
    assert_eq!(art.blob.len(), 4);
    assert_eq!(art.manifest.len(), 1);
    assert_eq!(art.manifest[0].mangled_name, "vector_add");
}

#[test]
fn manifest_json_roundtrip() {
    let meta = KernelMeta {
        mangled_name: "vector_add".into(),
        arg_kinds: vec![ArgKind::BufferF32, ArgKind::ScalarU32],
        shared_memory_bytes: 256,
        recommended_block: Some(ferrum_gpu_core::Dim3::new(256, 1, 1)),
        required_caps: BackendCaps::Cuda { min_sm_major: 5, min_sm_minor: 0 },
    };
    let json = serde_json::to_string(&meta).unwrap();
    let back: KernelMeta = serde_json::from_str(&json).unwrap();
    assert_eq!(back.mangled_name, meta.mangled_name);
    assert_eq!(back.shared_memory_bytes, 256);
}

#[test]
fn backend_id_is_u8_repr() {
    assert_eq!(BackendId::Cuda as u8, 1);
    assert_eq!(BackendId::Vulkan as u8, 2);
}
