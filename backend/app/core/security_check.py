#!/usr/bin/env python3
"""
Security Check: Scan for hardcoded secrets before GitHub upload
"""

import os
import re
from pathlib import Path

def check_file_for_secrets(file_path):
    """Check a single file for potential secrets"""
    secrets_found = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            
            patterns = {
                'API Key': r'AIza[0-9A-Za-z\-_]{35}',
                'Password': r'["\']password["\']\s*[:=]\s*["\'][^"\']+["\']',
                'Secret': r'["\']secret["\']\s*[:=]\s*["\'][^"\']+["\']',
                'Token': r'["\']token["\']\s*[:=]\s*["\'][^"\']+["\']',
            }
            
            for line_num, line in enumerate(lines, 1):
                for secret_type, pattern in patterns.items():
                    matches = re.finditer(pattern, line, re.IGNORECASE)
                    for match in matches:
                        # Skip if it's in a comment explaining things
                        if 'example' in line.lower() or 'your_' in line.lower():
                            continue
                        secrets_found.append({
                            'file': file_path,
                            'line': line_num,
                            'type': secret_type,
                            'text': match.group()[:50]
                        })
    except Exception as e:
        pass
    
    return secrets_found

def main():
    print("\n" + "="*60)
    print("GitHub Upload Security Check")
    print("="*60 + "\n")
    
    # Files to check
    python_files = list(Path('.').rglob('*.py'))
    config_files = [Path('config.py'), Path('config_production.py'), Path('.env.example')]
    
    all_files = [f for f in python_files if '__pycache__' not in str(f)]
    all_files.extend([f for f in config_files if f.exists()])
    
    print(f"🔍 Scanning {len(all_files)} files for secrets...\n")
    
    all_secrets = []
    for file_path in all_files:
        secrets = check_file_for_secrets(file_path)
        all_secrets.extend(secrets)
    
    if all_secrets:
        print("⚠️  POTENTIAL SECRETS FOUND:\n")
        
        for secret in all_secrets:
            print(f"❌ {secret['type']} in {secret['file']}:{secret['line']}")
            print(f"   {secret['text'][:80]}...")
            print()
        
        print(f"\n⚠️  Found {len(all_secrets)} potential secrets!")
        print("\n⚠️  ACTIONS REQUIRED:")
        print("   1. Move secrets to .env file")
        print("   2. Update code to read from environment variables")
        print("   3. Ensure .env is in .gitignore")
        print("   4. Re-run this check before uploading")
        
    else:
        print("✅ No hardcoded secrets detected!")
        print("\n📋 Additional Checks:")
    
    # Check .env exists and is in .gitignore
    env_exists = Path('.env').exists()
    gitignore_exists = Path('.gitignore').exists()
    
    if env_exists:
        print(f"   ⚠️  .env file exists - ensure it's in .gitignore")
    else:
        print(f"   ✅ No .env file found")
    
    if gitignore_exists:
        with open('.gitignore', 'r') as f:
            gitignore_content = f.read()
            if '.env' in gitignore_content:
                print(f"   ✅ .env is in .gitignore")
            else:
                print(f"   ⚠️  .env not found in .gitignore - ADD IT!")
    else:
        print(f"   ⚠️  No .gitignore file - CREATE ONE!")
    
    # Check for database files
    db_files = list(Path('.').rglob('*.db'))
    if db_files:
        print(f"   ⚠️  {len(db_files)} .db files found - ensure they're gitignored")
    else:
        print(f"   ✅ No .db files in project")
    
    print("\n" + "="*60)
    
    if not all_secrets and gitignore_exists:
        print("\n✅ SECURITY CHECK PASSED - Safe to upload!")
    else:
        print("\n⚠️  SECURITY CHECK FAILED - Fix issues before upload!")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
