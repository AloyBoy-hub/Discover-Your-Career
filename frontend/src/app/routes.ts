import { createBrowserRouter } from "react-router";
import { UploadPage } from "@/app/pages/UploadPage";
import { ResultsPage } from "@/app/pages/ResultsPage";
import { JobDetailPage } from "@/app/pages/JobDetailPage";
import { SurveyPage } from "@/app/pages/Survey";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: UploadPage,
  },
  {
    path: "/survey",
    Component: SurveyPage,
  },
  {
    path: "/results",
    Component: ResultsPage,
  },
  {
    path: "/job/:jobId",
    Component: JobDetailPage,
  },
]);
