"""
RK35xx 局域网设备 SSH 验收模块。

职责：
  - 通过 SSH/SFTP 连接局域网 RK35xx 设备（密码或私钥认证）。
  - 上传待验收的 .rknn 模型、设备端推理/验收脚本，以及验证图片（或直接引用设备上已有目录）。
  - 远程执行 device_validate.py，实时回传日志。
  - 下载生成的 JSON 验收报告到本地 output/device-validation/。

安全提示：
  - 默认信任目标主机公钥（AutoAddPolicy），适用于受控局域网内的开发/测试设备。
  - 不在日志、报告或异常信息中回显密码或私钥内容。
"""
import io
import os
import json
import posixpath
import shlex
import time
import hashlib
import logging

import paramiko

logger = logging.getLogger(__name__)

LOCAL_SCRIPT_FILES = ('infer_on_device.py', 'device_validate.py')
IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')


class DeviceSSHError(RuntimeError):
    """SSH 连接、传输或远程执行失败时抛出，消息不包含凭据。"""


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _build_pkey(key_text, key_passphrase):
    """从私钥文本构建 paramiko PKey，依次尝试常见格式。"""
    key_stream = io.StringIO(key_text)
    errors = []
    for key_cls in (paramiko.RSAKey, paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.DSSKey):
        try:
            key_stream.seek(0)
            return key_cls.from_private_key(key_stream, password=key_passphrase or None)
        except Exception as exc:  # noqa: BLE001 - 需要尝试多种格式
            errors.append(f'{key_cls.__name__}: {exc}')
    raise DeviceSSHError('无法解析私钥（已尝试 RSA/Ed25519/ECDSA/DSS）：' + '; '.join(errors))


def connect(host, port, username, password=None, key_text=None, key_passphrase=None, timeout=10):
    """建立 SSH 连接，返回已连接的 SSHClient。凭据不会被记录到日志。"""
    if not host or not username:
        raise DeviceSSHError('缺少设备地址或用户名')

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs = {
        'hostname': host,
        'port': int(port or 22),
        'username': username,
        'timeout': timeout,
        'banner_timeout': timeout,
        'auth_timeout': timeout,
    }
    if key_text:
        connect_kwargs['pkey'] = _build_pkey(key_text, key_passphrase)
    elif password:
        connect_kwargs['password'] = password
        connect_kwargs['allow_agent'] = False
        connect_kwargs['look_for_keys'] = False
    else:
        raise DeviceSSHError('未提供密码或私钥，无法认证')

    try:
        client.connect(**connect_kwargs)
    except paramiko.AuthenticationException as exc:
        raise DeviceSSHError('认证失败，请检查用户名/密码/私钥') from exc
    except (paramiko.SSHException, OSError) as exc:
        raise DeviceSSHError(f'连接失败：{exc}') from exc
    return client


def test_connection(host, port, username, password=None, key_text=None, key_passphrase=None):
    """快速测试连接并检查设备端 rknn-toolkit-lite2 可用性，不做任何文件传输。"""
    client = connect(host, port, username, password, key_text, key_passphrase)
    try:
        info = {'ok': True, 'message': '连接成功'}
        exit_code, out, err = _run_once(client, 'python3 --version')
        info['python_version'] = (out or err).strip()

        exit_code, out, err = _run_once(
            client, "python3 -c \"import rknnlite; print(rknnlite.__version__)\" 2>&1"
        )
        if exit_code == 0:
            info['rknnlite_version'] = out.strip()
        else:
            info['rknnlite_version'] = None
            info['rknnlite_warning'] = (out + err).strip()[:300]

        exit_code, out, err = _run_once(client, 'uname -a')
        info['uname'] = out.strip()
        npu_access = _check_npu_access(client, username, raise_on_error=False)
        info.update(npu_access)
        return info
    finally:
        client.close()


def _run_once(client, command, timeout=20):
    """执行一次性短命令，返回 (exit_code, stdout, stderr)。"""
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    out = stdout.read().decode('utf-8', errors='replace')
    err = stderr.read().decode('utf-8', errors='replace')
    exit_code = stdout.channel.recv_exit_status()
    return exit_code, out, err


def _check_npu_access(client, username, raise_on_error=True):
    """检查当前 SSH 用户是否可读写 RK NPU DRM/legacy 设备节点。"""
    command = (
        "found=0; "
        "for node in /dev/dri/by-path/*npu-render /dev/dri/by-path/*npu-card /dev/rknpu*; do "
        "[ -e \"$node\" ] || continue; found=1; "
        "if [ -r \"$node\" ] && [ -w \"$node\" ]; then echo \"OK:$node\"; exit 0; fi; "
        "echo \"DENIED:$node\"; "
        "done; [ \"$found\" -eq 1 ] && exit 13; echo MISSING; exit 12"
    )
    exit_code, out, _err = _run_once(client, command)
    result = {'npu_access': exit_code == 0, 'npu_device': None, 'npu_warning': None}
    if exit_code == 0:
        result['npu_device'] = out.strip().split(':', 1)[-1]
        return result

    if exit_code == 13:
        message = (
            f'用户 {username} 无权访问 RK NPU 设备节点。请在设备上执行 '
            f'`sudo usermod -aG video,render {username}`，然后注销并重新登录。'
        )
    else:
        message = '未发现 RK NPU 设备节点，请检查 RKNPU 驱动是否已加载以及设备树是否启用 NPU。'
    result['npu_warning'] = message
    if raise_on_error:
        raise DeviceSSHError(message)
    return result


def _stream_exec(client, command, log_cb=None, poll_interval=0.1):
    """执行长命令并实时把 stdout/stderr 按行推送给 log_cb，返回退出码。"""
    chan = client.get_transport().open_session()
    chan.exec_command(command)
    stdout_buf = b''
    stderr_buf = b''

    def _flush(buf, prefix=''):
        text = buf.decode('utf-8', errors='replace')
        lines = text.split('\n')
        remainder = lines.pop() if not text.endswith('\n') else ''
        for line in lines:
            if line.strip() and log_cb:
                log_cb(prefix + line)
        return remainder.encode('utf-8')

    while True:
        made_progress = False
        if chan.recv_ready():
            stdout_buf += chan.recv(4096)
            stdout_buf = _flush(stdout_buf)
            made_progress = True
        if chan.recv_stderr_ready():
            stderr_buf += chan.recv_stderr(4096)
            stderr_buf = _flush(stderr_buf, prefix='[stderr] ')
            made_progress = True
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if not made_progress:
            time.sleep(poll_interval)

    if stdout_buf and log_cb:
        log_cb(stdout_buf.decode('utf-8', errors='replace'))
    if stderr_buf and log_cb:
        log_cb('[stderr] ' + stderr_buf.decode('utf-8', errors='replace'))

    return chan.recv_exit_status()


def _sftp_mkdirs(sftp, remote_dir):
    """递归创建远程目录（类似 mkdir -p）。"""
    if not remote_dir or remote_dir in ('/', '.'):
        return
    try:
        sftp.stat(remote_dir)
        return
    except FileNotFoundError:
        pass
    parent = posixpath.dirname(remote_dir.rstrip('/'))
    if parent and parent != remote_dir:
        _sftp_mkdirs(sftp, parent)
    try:
        sftp.mkdir(remote_dir)
    except OSError:
        # 并发或已存在时忽略
        try:
            sftp.stat(remote_dir)
        except FileNotFoundError as exc:
            raise DeviceSSHError(f'创建远程目录失败：{remote_dir}') from exc


def _upload_file(sftp, local_path, remote_path):
    _sftp_mkdirs(sftp, posixpath.dirname(remote_path))
    sftp.put(local_path, remote_path)


def _upload_images_dir(sftp, local_dir, remote_dir, log_cb=None, log_every=25):
    count = 0
    for root, _dirs, files in os.walk(local_dir):
        rel = os.path.relpath(root, local_dir)
        remote_root = remote_dir if rel == '.' else posixpath.join(remote_dir, rel.replace('\\', '/'))
        image_files = [f for f in files if f.lower().endswith(IMAGE_EXTS)]
        if not image_files:
            continue
        _sftp_mkdirs(sftp, remote_root)
        for filename in image_files:
            sftp.put(os.path.join(root, filename), posixpath.join(remote_root, filename))
            count += 1
            if log_cb and count % log_every == 0:
                log_cb(f'▶ 已上传 {count} 张图片...')
    return count


def run_device_validation(
    *,
    host, port, username,
    password=None, key_text=None, key_passphrase=None,
    model_path,
    script_dir,
    images_mode,               # 'upload' | 'remote_path'
    local_images_dir=None,
    remote_images_path=None,
    remote_workdir='~/pt2rknn-device-validate',
    remote_python='python3',
    classes='',
    width=640, height=640,
    conf=0.001, iou=0.65, warmup=3,
    local_report_dir=None,
    log_cb=None,
):
    """
    在远程 RK35xx 设备上执行完整验收流程：连接 -> 上传 -> 执行 -> 下载报告。

    返回 dict：
      {
        'success': bool, 'message': str,
        'report': <解析后的 JSON 报告 dict 或 None>,
        'local_report_path': <本地保存路径或 None>,
        'model_sha256_local': str,
        'model_sha256_match': bool 或 None（无法比较时为 None）,
      }
    """
    def _log(msg):
        if log_cb:
            log_cb(msg)

    if not os.path.exists(model_path):
        raise DeviceSSHError(f'本地模型文件不存在：{model_path}')
    if images_mode == 'upload' and not (local_images_dir and os.path.isdir(local_images_dir)):
        raise DeviceSSHError('已选择上传图片模式，但本地图片目录不存在')
    if images_mode == 'remote_path' and not remote_images_path:
        raise DeviceSSHError('已选择设备已有目录模式，但未提供远程图片路径')

    model_sha256_local = _sha256_file(model_path)
    model_stem = os.path.splitext(os.path.basename(model_path))[0]
    remote_task_dir = posixpath.join(remote_workdir.rstrip('/'), model_stem)

    _log(f'▶ 连接设备 {username}@{host}:{port} ...')
    client = connect(host, port, username, password, key_text, key_passphrase)
    try:
        _log('✔ SSH 连接成功')
        npu_access = _check_npu_access(client, username)
        _log(f"✔ NPU 设备节点可访问：{npu_access['npu_device']}")
        sftp = client.open_sftp()

        # 展开远程 ~ 路径
        exit_code, out, _err = _run_once(client, 'echo $HOME')
        home_dir = out.strip() or '/root'
        if remote_task_dir.startswith('~'):
            remote_task_dir = home_dir + remote_task_dir[1:]

        _log(f'▶ 远程工作目录：{remote_task_dir}')
        _sftp_mkdirs(sftp, remote_task_dir)

        remote_model_path = posixpath.join(remote_task_dir, os.path.basename(model_path))
        _log('▶ 上传模型文件...')
        _upload_file(sftp, model_path, remote_model_path)

        for script_name in LOCAL_SCRIPT_FILES:
            local_script = os.path.join(script_dir, script_name)
            if not os.path.exists(local_script):
                raise DeviceSSHError(f'缺少设备端脚本：{local_script}')
            _log(f'▶ 上传脚本：{script_name}')
            _upload_file(sftp, local_script, posixpath.join(remote_task_dir, script_name))

        if images_mode == 'upload':
            remote_images_dir = posixpath.join(remote_task_dir, 'images')
            _log(f'▶ 上传验证图片：{local_images_dir} -> {remote_images_dir}')
            count = _upload_images_dir(sftp, local_images_dir, remote_images_dir, log_cb=_log)
            if count == 0:
                raise DeviceSSHError('本地图片目录中未找到图片')
            _log(f'✔ 图片上传完成，共 {count} 张')
        else:
            remote_images_dir = remote_images_path

        sftp.close()

        remote_report_path = posixpath.join(remote_task_dir, 'device-validation.json')
        classes_arg = classes if isinstance(classes, str) else ','.join(classes)
        command = (
            f"cd {shlex.quote(remote_task_dir)} && {shlex.quote(remote_python)} device_validate.py "
            f"--model {shlex.quote(posixpath.basename(remote_model_path))} "
            f"--images {shlex.quote(remote_images_dir)} "
            f"--classes {shlex.quote(classes_arg)} "
            f"--width {int(width)} --height {int(height)} "
            f"--conf {float(conf)} --iou {float(iou)} --warmup {int(warmup)} "
            f"--output device-validation.json"
        )
        _log(f'▶ 远程执行：{command}')
        exit_code = _stream_exec(client, command, log_cb=_log)

        if exit_code != 0:
            return {
                'success': False,
                'message': f'远程验收脚本执行失败，退出码 {exit_code}',
                'report': None,
                'local_report_path': None,
                'model_sha256_local': model_sha256_local,
                'model_sha256_match': None,
                'remote_task_dir': remote_task_dir,
            }

        sftp = client.open_sftp()
        try:
            sftp.stat(remote_report_path)
        except FileNotFoundError as exc:
            raise DeviceSSHError('远程验收完成但未生成报告文件') from exc

        report_dir = local_report_dir or os.path.join(os.path.dirname(model_path), 'device-validation')
        os.makedirs(report_dir, exist_ok=True)
        timestamp = int(time.time())
        local_report_path = os.path.join(report_dir, f'{model_stem}_{timestamp}.json')
        _log(f'▶ 下载报告到本地：{local_report_path}')
        sftp.get(remote_report_path, local_report_path)
        sftp.close()

        with open(local_report_path, 'r', encoding='utf-8') as file_obj:
            report = json.load(file_obj)

        remote_sha256 = (report.get('model') or {}).get('sha256')
        model_sha256_match = (remote_sha256 == model_sha256_local) if remote_sha256 else None
        if model_sha256_match is False:
            _log('⚠ 警告：设备端模型哈希与本地模型不一致，报告可能对应旧版本模型')
        elif model_sha256_match:
            _log('✔ 模型哈希校验一致')

        _log('✔ 设备验收完成')
        return {
            'success': True,
            'message': '设备验收完成',
            'report': report,
            'local_report_path': local_report_path,
            'model_sha256_local': model_sha256_local,
            'model_sha256_match': model_sha256_match,
            'remote_task_dir': remote_task_dir,
        }
    finally:
        client.close()
