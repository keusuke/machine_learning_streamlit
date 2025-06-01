import subprocess
import sys
import json
import datetime


def run_tests():
    """テストを実行し、結果を集計する"""
    print("="*60)
    print("テスト実行開始")
    print("="*60)
    
    start_time = datetime.datetime.now()
    
    # pytestコマンドを構築
    cmd = [
        sys.executable, "-m", "pytest",
        "tests/",
        "-v",
        "--tb=short",
        "--cov=src",
        "--cov-report=term-missing",
        "--cov-report=json",
        "-q"
    ]
    
    # テスト実行
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = datetime.datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    print(result.stdout)
    if result.stderr:
        print("エラー出力:")
        print(result.stderr)
    
    # カバレッジレポートを読み込む
    try:
        with open('coverage.json', 'r') as f:
            coverage_data = json.load(f)
            total_coverage = coverage_data['totals']['percent_covered']
    except:
        total_coverage = "N/A"
    
    # 結果の集計
    print("\n" + "="*60)
    print("テスト結果集計")
    print("="*60)
    
    # stdoutからテスト結果を解析
    lines = result.stdout.split('\n')
    passed = failed = 0
    
    for line in lines:
        if " passed" in line and " failed" not in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "passed" and i > 0:
                    try:
                        passed = int(parts[i-1])
                    except:
                        pass
        elif " failed" in line:
            parts = line.split()
            for i, part in enumerate(parts):
                if part == "failed" and i > 0:
                    try:
                        failed = int(parts[i-1])
                    except:
                        pass
    
    total_tests = passed + failed
    
    print(f"実行時間: {duration:.2f}秒")
    print(f"総テスト数: {total_tests}")
    print(f"成功: {passed}")
    print(f"失敗: {failed}")
    if total_tests > 0:
        print(f"成功率: {(passed/total_tests)*100:.1f}%")
    print(f"コードカバレッジ: {total_coverage}%")
    
    print("\n" + "="*60)
    
    if result.returncode == 0:
        print("✅ すべてのテストが成功しました！")
    else:
        print("❌ テストに失敗がありました")
    
    print("="*60)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)