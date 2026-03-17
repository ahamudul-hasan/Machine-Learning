#!/usr/bin/env python3
"""
Checklist Progress Auto-Updater
Automatically calculates and updates progress percentages in Checklist.md
"""

import re
from pathlib import Path
from datetime import datetime

class ChecklistUpdater:
    def __init__(self, checklist_path):
        self.checklist_path = Path(checklist_path)
        self.content = self.checklist_path.read_text(encoding='utf-8')
        self.sections = {
            '📚 Python Libraries': None,
            '📥 Data Gathering': None,
            '📊 Data Analysis': None,
            '🔧 Features Engineering': None,
            '🤖 Model Development': None,
        }
    
    def count_items(self, text):
        """Count completed and total items in a section"""
        completed = len(re.findall(r'\- \[X\]', text))
        total = len(re.findall(r'\- \[\s?[X ]?\]', text))
        return completed, total
    
    def extract_sections(self):
        """Extract each phase section and count items"""
        # Phase headers with numbers
        phase_headers = [
            ('📚 Python Libraries', r'## 📚 Phase 1:.*?(?=## 📥|## 📊|## 🔧|## 🤖|\Z)'),
            ('📥 Data Gathering', r'## 📥 Phase 2:.*?(?=## 📊|## 🔧|## 🤖|\Z)'),
            ('📊 Data Analysis', r'## 📊 Phase 3:.*?(?=## 🔧|## 🤖|\Z)'),
            ('🔧 Features Engineering', r'## 🔧 Phase 4:.*?(?=## 🤖|\Z)'),
            ('🤖 Model Development', r'## 🤖 Phase 5:.*?(?=\Z)'),
        ]
        
        for section_name, pattern in phase_headers:
            match = re.search(pattern, self.content, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(0)
                completed, total = self.count_items(content)
                self.sections[section_name] = {
                    'completed': completed,
                    'total': total,
                    'percentage': (completed / total * 100) if total > 0 else 0
                }
        
        return self.sections
    
    def calculate_totals(self):
        """Calculate overall totals"""
        total_completed = sum(s['completed'] for s in self.sections.values() if s)
        total_items = sum(s['total'] for s in self.sections.values() if s)
        overall_percentage = (total_completed / total_items * 100) if total_items > 0 else 0
        return total_completed, total_items, overall_percentage
    
    def get_status_emoji(self, percentage):
        """Get status emoji based on percentage"""
        if percentage == 100:
            return "✅ Complete"
        elif percentage > 0:
            return "🟠 In Progress"
        else:
            return "⏳ Not Started"
    
    def update_header(self):
        """Update the progress header and table"""
        self.extract_sections()
        total_completed, total_items, overall_percentage = self.calculate_totals()
        
        # Update overall progress line
        old_progress = r'> \*\*Overall Progress:\*\* 📊 \d+% Complete \(\d+/\d+ tasks\)'
        new_progress = f'> **Overall Progress:** 📊 {overall_percentage:.0f}% Complete ({total_completed}/{total_items} tasks)'
        self.content = re.sub(old_progress, new_progress, self.content)
        
        # Update table rows - more flexible pattern
        for section_name, data in self.sections.items():
            if data:
                percentage = data['percentage']
                status = self.get_status_emoji(percentage)
                # Match the section name in the table with flexible spacing and emoji
                section_emoji = section_name.split()[0]
                pattern = f"\\| {section_emoji}.*?\\| .+? \\| .+? \\|"
                replacement = f"| {section_name} | {status} | {percentage:.0f}% ({data['completed']}/{data['total']}) |"
                self.content = re.sub(pattern, replacement, self.content)
        
        # Update last modified date
        old_date = r'> \*\*Last Updated:\*\* .+'
        new_date = f'> **Last Updated:** {datetime.now().strftime("%B %d, %Y @ %H:%M")}'
        self.content = re.sub(old_date, new_date, self.content)
    
    def save(self):
        """Save updated content back to file"""
        self.checklist_path.write_text(self.content, encoding='utf-8')
        print(f"✅ Checklist updated successfully!")
        self._print_summary()
    
    def _print_summary(self):
        """Print progress summary"""
        total_completed, total_items, overall_percentage = self.calculate_totals()
        
        print("\n📊 Progress Summary:")
        print("=" * 50)
        for section_name in self.sections:
            data = self.sections[section_name]
            if data:
                bar_length = 20
                filled = int(bar_length * data['percentage'] / 100)
                bar = "█" * filled + "░" * (bar_length - filled)
                print(f"{section_name:25} {bar} {data['percentage']:5.1f}%")
        
        print("=" * 50)
        bar_length = 20
        filled = int(bar_length * overall_percentage / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        print(f"{'OVERALL':25} {bar} {overall_percentage:5.1f}%")
        print(f"\nTotal: {total_completed}/{total_items} tasks completed")


def main():
    checklist_path = Path(__file__).parent / "Checklist.md"
    
    if not checklist_path.exists():
        print(f"❌ Error: Checklist.md not found at {checklist_path}")
        return
    
    print("🔄 Updating checklist progress...\n")
    updater = ChecklistUpdater(checklist_path)
    updater.update_header()
    updater.save()


if __name__ == "__main__":
    main()
