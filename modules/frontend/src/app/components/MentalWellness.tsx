import { Activity, BookOpen, HeartPulse } from 'lucide-react';

export function MentalWellness() {
  const menuItems = [
    { icon: Activity, label: 'Activity Feed', active: true },
    { icon: BookOpen, label: 'Journal', active: false },
    { icon: HeartPulse, label: 'Resources', active: false },
  ];

  return (
    <div className="bg-[#2A2A2A] rounded-lg p-4">
      <h3 className="text-white font-semibold mb-3">Mental Wellness</h3>

      <div className="space-y-2">
        {menuItems.map((item) => (
          <button
            key={item.label}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors ${
              item.active
                ? 'bg-blue-600 text-white'
                : 'bg-[#1A1A1A] text-gray-400 hover:bg-[#252525]'
            }`}
          >
            <item.icon size={18} />
            <span className="text-sm font-medium">{item.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
