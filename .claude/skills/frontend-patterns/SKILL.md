---
name: frontend-patterns
description: Design patterns for React/TypeScript frontends. Apply for component design, API communication, and Vite configuration.
allowed-tools: Read, Grep, Glob
---

# Frontend Patterns

## Component Design

### File Layout
```
src/
  components/     # Reusable UI components
  pages/          # Page components
  hooks/          # Custom hooks
  utils/          # Utility functions
  types/          # Type definitions
```

### Component Template
```tsx
interface Props {
  title: string;
  onSubmit: (data: FormData) => void;
  isLoading?: boolean;
}

export const MyComponent: React.FC<Props> = ({ title, onSubmit, isLoading = false }) => {
  const [value, setValue] = useState("");

  const handleSubmit = useCallback(() => {
    onSubmit({ value });
  }, [value, onSubmit]);

  return (
    <div>
      <h2>{title}</h2>
      <button onClick={handleSubmit} disabled={isLoading}>
        {isLoading ? "Processing..." : "Submit"}
      </button>
    </div>
  );
};
```

## API Communication

### fetch Wrapper
```typescript
const apiCall = async <T>(url: string, options?: RequestInit): Promise<T> => {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`API Error: ${response.status}`);
  return response.json();
};
```

### Loading / Error State
```typescript
const [data, setData] = useState<Data | null>(null);
const [error, setError] = useState<string | null>(null);
const [isLoading, setIsLoading] = useState(false);
```

## Vite Configuration Notes

- `npm run dev` and `npm run preview` behave differently with `publicDir`
- For production-equivalent verification: `npm run build && npm run preview`
- Place `config.json` under `public/` → it is copied into `dist/` after build
