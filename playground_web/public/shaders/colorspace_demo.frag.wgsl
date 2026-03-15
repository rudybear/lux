var<private> uv_1: vec2<f32>;
var<private> color: vec4<f32>;

fn main_1() {
    let _e20 = uv_1;
    let _e22 = uv_1;
    let _e26 = uv_1;
    let _e30 = vec3<f32>(_e20.x, (0.7f + (_e22.y * 0.3f)), (0.5f + (_e26.y * 0.5f)));
    let _e32 = (_e30.x * 6f);
    let _e35 = (_e30.z * _e30.y);
    let _e43 = (_e35 * (1f - abs(((_e32 - (floor((_e32 / 2f)) * 2f)) - 1f))));
    let _e44 = (_e30.z - _e35);
    let _e45 = (_e32 < 1f);
    let _e46 = (_e32 < 2f);
    let _e47 = (_e32 < 3f);
    let _e48 = (_e32 < 4f);
    let _e49 = (_e32 < 5f);
    let _e68 = vec3<f32>((select(select(select(select(select(_e35, _e43, _e49), 0f, _e48), 0f, _e47), _e43, _e46), _e35, _e45) + _e44), (select(select(select(select(select(0f, 0f, _e49), _e43, _e48), _e35, _e47), _e35, _e46), _e43, _e45) + _e44), (select(select(select(select(select(_e43, _e35, _e49), _e35, _e48), _e43, _e47), 0f, _e46), 0f, _e45) + _e44));
    let _e69 = uv_1;
    let _e71 = (_e69.y * 3f);
    let _e75 = mix(vec3(dot(_e68, vec3<f32>(0.2126f, 0.7152f, 0.0722f))), _e68, vec3(1.5f));
    let _e80 = max(max(_e75.x, _e75.y), _e75.z);
    let _e83 = (_e80 - min(min(_e75.x, _e75.y), _e75.z));
    let _e87 = ((_e75.y - _e75.z) / _e83);
    let _e106 = vec3<f32>((select(select(select((((_e75.x - _e75.y) / _e83) + 4f), (((_e75.z - _e75.x) / _e83) + 2f), (_e80 == _e75.y)), (_e87 - (floor((_e87 / 6f)) * 6f)), (_e80 == _e75.x)), 0f, (_e83 < 0.00001f)) / 6f), select((_e83 / _e80), 0f, (_e80 < 0.00001f)), _e80);
    let _e108 = (_e106.x + 0.15f);
    let _e113 = vec3(0.5f);
    let _e121 = vec3<f32>((_e108 - (floor((_e108 / 1f)) * 1f)), _e106.y, _e106.z);
    let _e123 = (_e121.x * 6f);
    let _e126 = (_e121.z * _e121.y);
    let _e134 = (_e126 * (1f - abs(((_e123 - (floor((_e123 / 2f)) * 2f)) - 1f))));
    let _e135 = (_e121.z - _e126);
    let _e136 = (_e123 < 1f);
    let _e137 = (_e123 < 2f);
    let _e138 = (_e123 < 3f);
    let _e139 = (_e123 < 4f);
    let _e140 = (_e123 < 5f);
    let _e164 = select(select(pow(_e68, vec3((1f / 2.2f))), vec3<f32>((select(select(select(select(select(_e126, _e134, _e140), 0f, _e139), 0f, _e138), _e134, _e137), _e126, _e136) + _e135), (select(select(select(select(select(0f, 0f, _e140), _e134, _e139), _e126, _e138), _e126, _e137), _e134, _e136) + _e135), (select(select(select(select(select(_e134, _e126, _e140), _e126, _e139), _e134, _e138), 0f, _e137), 0f, _e136) + _e135)), (_e71 < 2f)), (((_e68 - _e113) * 1.5f) + _e113), (_e71 < 1f));
    let _e169 = max(max(_e164.x, _e164.y), _e164.z);
    let _e172 = (_e169 - min(min(_e164.x, _e164.y), _e164.z));
    let _e176 = ((_e164.y - _e164.z) / _e172);
    let _e195 = vec3<f32>((select(select(select((((_e164.x - _e164.y) / _e172) + 4f), (((_e164.z - _e164.x) / _e172) + 2f), (_e169 == _e164.y)), (_e176 - (floor((_e176 / 6f)) * 6f)), (_e169 == _e164.x)), 0f, (_e172 < 0.00001f)) / 6f), select((_e172 / _e169), 0f, (_e169 < 0.00001f)), _e169);
    let _e197 = (_e195.x * 6f);
    let _e200 = (_e195.z * _e195.y);
    let _e208 = (_e200 * (1f - abs(((_e197 - (floor((_e197 / 2f)) * 2f)) - 1f))));
    let _e209 = (_e195.z - _e200);
    let _e210 = (_e197 < 1f);
    let _e211 = (_e197 < 2f);
    let _e212 = (_e197 < 3f);
    let _e213 = (_e197 < 4f);
    let _e214 = (_e197 < 5f);
    let _e234 = uv_1;
    let _e237 = select(vec3<f32>((select(select(select(select(select(_e200, _e208, _e214), 0f, _e213), 0f, _e212), _e208, _e211), _e200, _e210) + _e209), (select(select(select(select(select(0f, 0f, _e214), _e208, _e213), _e200, _e212), _e200, _e211), _e208, _e210) + _e209), (select(select(select(select(select(_e208, _e200, _e214), _e200, _e213), _e208, _e212), 0f, _e211), 0f, _e210) + _e209)), _e164, (_e234.x < 0.5f));
    color = vec4<f32>(_e237.x, _e237.y, _e237.z, 1f);
    return;
}

@fragment 
fn main(@location(0) uv: vec2<f32>) -> @location(0) vec4<f32> {
    uv_1 = uv;
    main_1();
    let _e3 = color;
    return _e3;
}
