from typing import *
from pydantic_model import \
    SequenceKey  # 假设这个pydantic模型定义为: class SequenceKey(BaseModel): frame: int; location: Tuple[float,float,float]; rotation: Tuple[float,float,float]

import unreal
from utils import *
from utils_actor import *
import math
import csv


################################################################################
# misc (这部分代码无需修改，保持原样)
def convert_frame_rate_to_fps(frame_rate: unreal.FrameRate) -> float:
    return frame_rate.numerator / frame_rate.denominator


def get_sequence_fps(sequence: unreal.LevelSequence) -> float:
    seq_fps: unreal.FrameRate = sequence.get_display_rate()
    return convert_frame_rate_to_fps(seq_fps)


def get_animation_length(animation_asset: unreal.AnimSequence, seq_fps: Optional[float] = None) -> int:
    anim_len = animation_asset.get_editor_property("number_of_sampled_frames")
    if seq_fps:
        anim_frame_rate = animation_asset.get_editor_property("target_frame_rate")
        anim_frame_rate = convert_frame_rate_to_fps(anim_frame_rate)
        if anim_frame_rate != seq_fps:
            anim_len = round(animation_asset.get_editor_property("sequence_length") * seq_fps)
    return anim_len


################################################################################
# sequencer session (这部分代码也无需修改，保持原样)
# ... 从你的脚本中复制 get_transform_channels_from_section 到 get_spawnable_actor_from_binding 的所有函数 ...
# ... (此处省略了所有未修改的底层函数，以保持简洁) ...
# 为了完整性，我将它们粘贴在下面，你实际上不需要改动它们

def get_transform_channels_from_section(
        trans_section: unreal.MovieScene3DTransformSection,
) -> List[unreal.MovieSceneScriptingChannel]:
    channel_x = channel_y = channel_z = channel_roll = channel_pitch = channel_yaw = None
    for channel in trans_section.get_channels():
        channel: unreal.MovieSceneScriptingChannel
        if channel.channel_name == "Location.X":
            channel_x = channel
        elif channel.channel_name == "Location.Y":
            channel_y = channel
        elif channel.channel_name == "Location.Z":
            channel_z = channel
        elif channel.channel_name == "Rotation.X":
            channel_roll = channel
        elif channel.channel_name == "Rotation.Y":
            channel_pitch = channel
        elif channel.channel_name == "Rotation.Z":
            channel_yaw = channel
    assert channel_x is not None and channel_y is not None and channel_z is not None
    assert channel_roll is not None and channel_pitch is not None and channel_yaw is not None
    return channel_x, channel_y, channel_z, channel_roll, channel_pitch, channel_yaw


def set_transforms_by_section(
        trans_section: unreal.MovieScene3DTransformSection,
        trans_keys: List[SequenceKey],
        key_type: str = "CONSTANT",
) -> None:
    channel_x, channel_y, channel_z, \
        channel_roll, channel_pitch, channel_yaw = get_transform_channels_from_section(trans_section)
    key_type_ = getattr(unreal.MovieSceneKeyInterpolation, key_type)

    for trans_key in trans_keys:
        key_frame = trans_key.frame
        loc_x, loc_y, loc_z = trans_key.location
        # UE Rotation (X,Y,Z) -> (Roll, Pitch, Yaw)
        rot_x, rot_y, rot_z = trans_key.rotation

        key_time_ = unreal.FrameNumber(key_frame)
        channel_x.add_key(key_time_, loc_x, interpolation=key_type_)
        channel_y.add_key(key_time_, loc_y, interpolation=key_type_)
        channel_z.add_key(key_time_, loc_z, interpolation=key_type_)
        channel_roll.add_key(key_time_, rot_x, interpolation=key_type_)
        channel_pitch.add_key(key_time_, rot_y, interpolation=key_type_)
        channel_yaw.add_key(key_time_, rot_z, interpolation=key_type_)


def add_transforms_to_binding(
        binding: unreal.SequencerBindingProxy,
        actor_trans_keys: List[SequenceKey],
        key_type: str = "CONSTANT",
) -> unreal.MovieScene3DTransformTrack:
    transform_track: unreal.MovieScene3DTransformTrack = binding.add_track(unreal.MovieScene3DTransformTrack)
    transform_section: unreal.MovieScene3DTransformSection = transform_track.add_section()
    transform_section.set_start_frame_bounded(False)  # 允许在序列范围外有关键帧
    transform_section.set_end_frame_bounded(False)
    set_transforms_by_section(transform_section, actor_trans_keys, key_type)
    return transform_track


def get_spawnable_actor_from_binding(sequence: unreal.MovieSceneSequence,
                                     binding: unreal.SequencerBindingProxy) -> unreal.Actor:
    binds = unreal.Array(unreal.SequencerBindingProxy)
    binds.append(binding)
    bound_objects: List[unreal.SequencerBoundObjects] = unreal.SequencerTools.get_bound_objects(
        get_world(), sequence, binds, sequence.get_playback_range()
    )
    return bound_objects[0].bound_objects[0]


def add_property_float_track_to_binding(
        binding: unreal.SequencerBindingProxy,
        property_name: str,
        property_value: float,
        float_track: Optional[unreal.MovieSceneFloatTrack] = None,
) -> unreal.MovieSceneFloatTrack:
    if float_track is None:
        float_track: unreal.MovieSceneFloatTrack = binding.add_track(unreal.MovieSceneFloatTrack)
        float_track.set_property_name_and_path(property_name, property_name)

    float_section = float_track.add_section()
    float_section.set_start_frame_bounded(0)
    float_section.set_end_frame_bounded(0)

    for channel in float_section.find_channels_by_type(unreal.MovieSceneScriptingFloatChannel):
        channel.set_default(property_value)
    return float_track


################################################################################
# high level functions (部分函数保持不变)

def add_spawnable_camera_to_sequence(
        sequence: unreal.LevelSequence,
        camera_trans: List[SequenceKey],
        camera_class: Type[unreal.CameraActor] = unreal.CameraActor,
        camera_fov: float = 90.,
        seq_length: Optional[int] = None,
        key_type: str = "CONSTANT",
) -> None:
    if seq_length is None:
        seq_length = sequence.get_playback_end()

    # 1. 创建相机 actor 的 spawnable 绑定
    camera_binding = sequence.add_spawnable_from_class(camera_class)

    # 2. 为相机组件的 'FieldOfView' 属性添加轨道
    # 这是更稳定和推荐的方式
    try:
        # 获取 CineCameraComponent 的模板对象
        spawnable_object_template = camera_binding.get_object_template()
        camera_component_template = spawnable_object_template.get_cine_camera_component()

        # 添加属性轨道
        fov_track = camera_binding.add_track(unreal.MovieSceneFloatTrack)

        # 将轨道绑定到 CineCameraComponent 的 FieldOfView 属性
        # 格式是: 'ComponentName.PropertyName'
        fov_track.set_property_name_and_path("CineCameraComponent.FieldOfView", "FieldOfView")

        # 为轨道添加一个 Section
        fov_section = fov_track.add_section()
        fov_section.set_start_frame_bounded(False)
        fov_section.set_end_frame_bounded(False)

        # 设置默认值
        fov_channel = fov_section.get_channels()[0]
        fov_channel.set_default(camera_fov)

    except Exception as e:
        unreal.log_warning(f"无法设置相机FOV轨道，将使用默认值。错误: {e}")

    # 3. 添加相机剪辑轨道 (Camera Cut Track)
    camera_cut_track = sequence.add_master_track(unreal.MovieSceneCameraCutTrack)
    camera_cut_section = camera_cut_track.add_section()
    camera_cut_section.set_range(0, seq_length)

    camera_binding_id = sequence.make_binding_id(camera_binding, unreal.MovieSceneObjectBindingSpace.LOCAL)
    camera_cut_section.set_camera_binding_id(camera_binding_id)

    # 4. 设置相机的位置和旋转
    add_transforms_to_binding(camera_binding, camera_trans, key_type)


def generate_sequence(
        sequence_dir: str,
        sequence_name: str,
        seq_fps: float,
        seq_length: int,
) -> unreal.LevelSequence:
    asset_tools: unreal.AssetTools = unreal.AssetToolsHelpers.get_asset_tools()
    factory = unreal.LevelSequenceFactoryNew()
    # 确保路径存在
    if not unreal.EditorAssetLibrary.does_directory_exist(sequence_dir):
        unreal.EditorAssetLibrary.make_directory(sequence_dir)

    new_sequence: unreal.LevelSequence = asset_tools.create_asset(
        sequence_name,
        sequence_dir,
        unreal.LevelSequence,
        factory,
    )
    assert (new_sequence is not None), f"Failed to create LevelSequence: {sequence_dir}/{sequence_name}"

    new_sequence.set_display_rate(unreal.FrameRate(numerator=int(seq_fps), denominator=1))
    new_sequence.set_playback_start(0)
    new_sequence.set_playback_end(seq_length)
    return new_sequence


# --- vvvvvvvvvv 这是主要修改的函数 vvvvvvvvvv ---
def generate_path_from_csv(
        csv_path: str,
        start_frame: int = 0
) -> Tuple[List[SequenceKey], int]:
    """
    从包含完整位姿的CSV文件读取数据，并为每一行生成一个关键帧。

    Args:
        csv_path (str): CSV文件的路径。文件应包含'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'列。
        start_frame (int): 轨迹的起始帧。

    Returns:
        Tuple[List[SequenceKey], int]: 返回生成的相机关键帧列表和最后的帧数。
    """
    camera_trans = []

    try:
        with open(csv_path, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            if reader.fieldnames is None:
                unreal.log_error(f"CSV文件 '{csv_path}' 为空或无法读取表头。")
                return [], start_frame

            required_columns = {'X', 'Y', 'Z', 'Yaw', 'Pitch', 'Roll'}
            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)
                unreal.log_error(f"CSV文件 '{csv_path}' 缺少必要的列: {', '.join(missing)}")
                return [], start_frame

            for i, row in enumerate(reader):
                location = (float(row['X']), float(row['Y']), float(row['Z']))
                rotation = (float(row['Roll']), float(row['Pitch']), float(row['Yaw']))

                camera_trans.append(
                    SequenceKey(
                        frame=start_frame + i,
                        location=location,
                        rotation=rotation
                    )
                )

    except FileNotFoundError:
        unreal.log_error(f"CSV文件未找到: {csv_path}")
        return [], start_frame
    except (ValueError, TypeError) as e:
        unreal.log_error(f"解析CSV数据时出错 (在行 {i + 2} 附近)，请检查数据格式是否为数字: {e}")
        return [], start_frame
    except Exception as e:
        unreal.log_error(f"读取或解析CSV文件时出错: {e}")
        return [], start_frame

    if not camera_trans:
        unreal.log_warning(f"CSV文件 '{csv_path}' 为空或未能成功解析任何行。")
        return [], start_frame

    last_frame = start_frame + len(camera_trans) - 1
    return camera_trans, last_frame


# --- ^^^^^^^^^^ 这是主要修改的函数 ^^^^^^^^^^ ---


# --- vvvvvvvvvv 这是主要修改的函数 vvvvvvvvvv ---
def main():
    # --- 1. 用户配置区域 ---
    csv_file_path = 'F:\CitySample\Plugins\MatrixCityPlugin\Content\Python\drone_stations_250.csv'  # !!! 请使用绝对路径 !!!
    sequence_dir = '/Game/Sequences/NightTime'
    sequence_name = 'Oblique_Sequence_250'
    seq_fps = 30.0
    fov = 45.0

    # --- 结束配置 ---

    unreal.log(f"开始执行相机轨迹导入脚本...")
    current_frame = 0

    # 2. 从CSV生成相机轨迹 (包含精确位姿)
    unreal.log(f"正在从CSV文件读取完整位姿路径: {csv_file_path}")
    camera_trans, last_frame = generate_path_from_csv(
        csv_path=csv_file_path,
        start_frame=current_frame
    )

    if not camera_trans:
        unreal.log_error("未能从CSV生成相机轨迹，脚本终止。")
        return

    # 序列总长应为关键帧数量，因为每帧一个关键点
    seq_length = len(camera_trans)
    unreal.log(f"轨迹生成完毕。总共 {len(camera_trans)} 个关键帧, 序列总长度: {seq_length} 帧。")

    # 3. 创建序列并添加相机
    new_sequence = generate_sequence(sequence_dir, sequence_name, seq_fps, seq_length)

    add_spawnable_camera_to_sequence(
        new_sequence,
        camera_trans=camera_trans,
        camera_class=unreal.CineCameraActor,  # 使用电影相机
        camera_fov=fov,
        seq_length=seq_length,
        key_type="CONSTANT"  # 确保每个关键帧都是阶梯式的
    )

    unreal.EditorAssetLibrary.save_loaded_asset(new_sequence)
    unreal.log(f"成功创建并保存序列: {sequence_dir}/{sequence_name}")

    return None, f'{sequence_dir}/{sequence_name}'


# --- ^^^^^^^^^^ 这是主要修改的函数 ^^^^^^^^^^ ---

if __name__ == "__main__":
    main()

