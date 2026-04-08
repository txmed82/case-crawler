import type { CredibilityLevel } from '../api/client';

interface SourceBadgeProps {
  name: string;
  credibilityLevel: CredibilityLevel;
  available?: boolean;
}

const credibilityColors: Record<CredibilityLevel, string> = {
  high: 'bg-green-100 text-green-800 border border-green-300',
  medium: 'bg-yellow-100 text-yellow-800 border border-yellow-300',
  low: 'bg-orange-100 text-orange-800 border border-orange-300',
  unknown: 'bg-gray-100 text-gray-600 border border-gray-300',
};

const credibilityLabels: Record<CredibilityLevel, string> = {
  high: 'High',
  medium: 'Medium',
  low: 'Low',
  unknown: 'Unknown',
};

export default function SourceBadge({ name, credibilityLevel, available = true }: SourceBadgeProps) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium ${credibilityColors[credibilityLevel]} ${!available ? 'opacity-50' : ''}`}
      title={`Credibility: ${credibilityLabels[credibilityLevel]}`}
    >
      <span
        className={`h-1.5 w-1.5 rounded-full ${
          credibilityLevel === 'high'
            ? 'bg-green-500'
            : credibilityLevel === 'medium'
            ? 'bg-yellow-500'
            : credibilityLevel === 'low'
            ? 'bg-orange-500'
            : 'bg-gray-400'
        }`}
      />
      {name}
    </span>
  );
}
