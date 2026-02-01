import { RouterProvider } from 'react-router';
import { router } from '@/app/routes';
import { FormProvider } from '@/app/context/FormContext';

export default function App() {
  return (
    <FormProvider>
      <RouterProvider router={router} />
    </FormProvider>
  );
}
