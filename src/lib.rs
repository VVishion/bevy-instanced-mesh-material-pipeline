use std::{hash::Hash, marker::PhantomData};

use bevy::{
    asset::load_internal_asset,
    core_pipeline::core_3d::{AlphaMask3d, Opaque3d, Transparent3d},
    ecs::query::QueryItem,
    pbr::{MaterialPipelineKey, MeshPipelineKey, MeshUniform, RenderMaterials},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        render_asset::RenderAssets,
        render_phase::{AddRenderCommand, DrawFunctions, RenderPhase},
        render_resource::*,
        renderer::RenderDevice,
        view::ExtractedView,
        RenderApp, RenderSet,
    },
};
use bytemuck::{Pod, Zeroable};
use pipeline::{DrawMeshInstancedWithMaterial, InstancedMeshMaterialPipeline};

use crate::pipeline::INSTANCED_MESH_SHADER_HANDLE;

pub mod pipeline;

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct Instance {
    position: Vec3,
}

#[derive(Component, Deref)]
pub struct Instances(pub Vec<Instance>);

impl ExtractComponent for Instances {
    type Query = &'static Self;
    type Filter = ();
    type Out = Self;

    fn extract_component(instances: QueryItem<'_, Self::Query>) -> Option<Self> {
        Some(Instances(instances.0.clone()))
    }
}

#[derive(Default)]
pub struct InstancedMeshMaterialPipelinePlugin<M> {
    marker: PhantomData<M>,
}

impl<M> Plugin for InstancedMeshMaterialPipelinePlugin<M>
where
    M: Material + Sync + Send + 'static,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    fn build(&self, app: &mut App) {
        load_internal_asset!(
            app,
            INSTANCED_MESH_SHADER_HANDLE,
            "instanced_mesh.wgsl",
            Shader::from_wgsl
        );

        app.add_plugin(ExtractComponentPlugin::<Instances>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Opaque3d, DrawMeshInstancedWithMaterial<M>>()
            .add_render_command::<AlphaMask3d, DrawMeshInstancedWithMaterial<M>>()
            .add_render_command::<Transparent3d, DrawMeshInstancedWithMaterial<M>>()
            .init_resource::<InstancedMeshMaterialPipeline<M>>()
            .init_resource::<SpecializedMeshPipelines<InstancedMeshMaterialPipeline<M>>>()
            .add_system(queue_instanced_meshes_with_material::<M>.in_set(RenderSet::Queue))
            .add_system(prepare_instance_buffers.in_set(RenderSet::Prepare));
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &Instances)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instances) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance data buffer"),
            contents: bytemuck::cast_slice(instances.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instances.len(),
        });
    }
}

#[allow(clippy::too_many_arguments)]
fn queue_instanced_meshes_with_material<M>(
    opaque_draw_functions: Res<DrawFunctions<Opaque3d>>,
    alpha_mask_draw_functions: Res<DrawFunctions<AlphaMask3d>>,
    transparent_draw_functions: Res<DrawFunctions<Transparent3d>>,
    instanced_mesh_material_pipeline: Res<InstancedMeshMaterialPipeline<M>>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<InstancedMeshMaterialPipeline<M>>>,
    pipeline_cache: Res<PipelineCache>,
    render_meshes: Res<RenderAssets<Mesh>>,
    render_materials: Res<RenderMaterials<M>>,
    instanced_meshes_with_material: Query<
        (Entity, &MeshUniform, &Handle<Mesh>, &Handle<M>),
        With<Instances>,
    >,
    mut views: Query<(
        &ExtractedView,
        &mut RenderPhase<Opaque3d>,
        &mut RenderPhase<AlphaMask3d>,
        &mut RenderPhase<Transparent3d>,
    )>,
) where
    M: Material,
    M::Data: PartialEq + Eq + Hash + Clone,
{
    let draw_instanced_mesh_with_opaque_material = opaque_draw_functions
        .read()
        .id::<DrawMeshInstancedWithMaterial<M>>();
    let draw_instanced_mesh_with_alpha_mask_material = alpha_mask_draw_functions
        .read()
        .id::<DrawMeshInstancedWithMaterial<M>>();
    let draw_instanced_mesh_with_transparent_material = transparent_draw_functions
        .read()
        .id::<DrawMeshInstancedWithMaterial<M>>();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples());

    for (view, mut opaque_phase, mut alpha_mask_phase, mut transparent_phase) in &mut views {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, mesh_uniform, mesh_handle, material_handle) in &instanced_meshes_with_material
        {
            if let (Some(mesh), Some(material)) = (
                render_meshes.get(mesh_handle),
                render_materials.get(material_handle),
            ) {
                let mut mesh_key =
                    MeshPipelineKey::from_primitive_topology(mesh.primitive_topology) | view_key;
                let alpha_mode = material.properties.alpha_mode;
                if let AlphaMode::Blend | AlphaMode::Premultiplied | AlphaMode::Add = alpha_mode {
                    mesh_key |= MeshPipelineKey::BLEND_PREMULTIPLIED_ALPHA;
                } else if let AlphaMode::Multiply = alpha_mode {
                    mesh_key |= MeshPipelineKey::BLEND_MULTIPLY;
                }

                let pipeline = pipelines
                    .specialize(
                        &pipeline_cache,
                        &instanced_mesh_material_pipeline,
                        MaterialPipelineKey {
                            mesh_key,
                            bind_group_data: material.key.clone(),
                        },
                        &mesh.layout,
                    )
                    .unwrap();

                let distance =
                    rangefinder.distance(&mesh_uniform.transform) + material.properties.depth_bias;

                let alpha_mode = material.properties.alpha_mode;

                match alpha_mode {
                    AlphaMode::Opaque => {
                        opaque_phase.add(Opaque3d {
                            entity,
                            draw_function: draw_instanced_mesh_with_opaque_material,
                            pipeline,
                            distance,
                        });
                    }
                    AlphaMode::Mask(_) => {
                        alpha_mask_phase.add(AlphaMask3d {
                            entity,
                            draw_function: draw_instanced_mesh_with_alpha_mask_material,
                            pipeline,
                            distance,
                        });
                    }
                    AlphaMode::Blend
                    | AlphaMode::Premultiplied
                    | AlphaMode::Add
                    | AlphaMode::Multiply => {
                        transparent_phase.add(Transparent3d {
                            entity,
                            draw_function: draw_instanced_mesh_with_transparent_material,
                            pipeline,
                            distance,
                        });
                    }
                }
            }
        }
    }
}
